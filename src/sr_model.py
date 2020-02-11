import os
import warnings
import torch
import datetime
import torch.optim as optim
import numpy as np
import sys

from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage import io, img_as_uint
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from collator import CollateFn
from dataloader import ImgDataset, ImgSet
from fusion_model import FusionModel
from registration_model import RegistrationModel
from utils import set_random_seed, get_imgset_dirs, load_baseline_cpsnr

from utils import shift_cpsnr, get_crop_mask, benchmark, get_sr_and_score
from zipfile import ZipFile

FUS_MODEL = 'fus_model.pth'
REG_MODEL = 'reg_model.pth'


class SrModel:
    """
    Super Resolved Model
    """

    def __init__(self, config):
        set_random_seed()

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fus_model = FusionModel()
        self.reg_model = RegistrationModel()

        self.optimizer = optim.Adam(list(self.fus_model.parameters()) + list(self.reg_model.parameters()),
                                    lr=config["training"]["lr"])  # optim
        data_directory = config["folder"]["data_dir"]
        self.baseline_cpsnrs = None
        if os.path.exists(os.path.join(data_directory, "norm.csv")):
            self.baseline_cpsnrs = load_baseline_cpsnr(os.path.join(data_directory, "norm.csv"))
        train_set_directories = get_imgset_dirs(os.path.join(data_directory, "train"))
        val_proportion = config['training']['val_proportion']
        train_list, val_list = train_test_split(train_set_directories,
                                                test_size=val_proportion,
                                                random_state=1, shuffle=True)
        # Dataloaders
        batch_size = config["training"]["batch_size"]
        n_workers = config["training"]["n_workers"]
        seq_len = config["training"]["seq_len"]

        train_dataset = ImgDataset(imgset_dir=train_list,
                                   seq_len=seq_len)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=n_workers,
                                      collate_fn=CollateFn(seq_len=seq_len),
                                      pin_memory=True)

        val_dataset = ImgDataset(imgset_dir=val_list,
                                 seq_len=seq_len)
        val_dataloader = DataLoader(val_dataset, batch_size=1,
                                    shuffle=False, num_workers=n_workers,
                                    collate_fn=CollateFn(seq_len=seq_len),
                                    pin_memory=True)

        self.dataloaders = {'train': train_dataloader, 'val': val_dataloader}

        torch.cuda.empty_cache()

    def train(self):
        """
        train SrModel
        """
        config = self.config
        num_epochs = config["training"]["num_epochs"]
        batch_size = config["training"]["batch_size"]
        seq_len = config["training"]["seq_len"]

        subfolder_pattern = 'batch_{}_seq_len_{}_time_{}'.format(
            batch_size, seq_len, f"{datetime.datetime.now():%Y-%m-%d-%H-%M}")

        checkpoint_dir_run = os.path.join(config["folder"]["checkpoints_dir"], subfolder_pattern)
        os.makedirs(checkpoint_dir_run, exist_ok=True)

        logs_dir = config['folder']['logs_dir']
        logging_dir = os.path.join(logs_dir, subfolder_pattern)
        os.makedirs(logging_dir, exist_ok=True)

        writer = SummaryWriter(logging_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        best_score = sys.maxsize
        torch_mask = None

        self.fus_model.to(device)
        self.reg_model.to(device)

        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=config['training']['lr_decay'],
                                                   verbose=True, patience=config['training']['lr_step'])

        for epoch in tqdm(range(1, num_epochs + 1)):
            self.fus_model.train()
            self.reg_model.train()
            train_loss = 0.0
            # Iterate over data.
            for lrs, hrs, hr_maps, names in tqdm(self.dataloaders['train']):
                if torch_mask is None:
                    patch_size = lrs.size(2)
                    crop_size = 3
                    torch_mask = get_crop_mask(patch_size, crop_size)
                    torch_mask = torch_mask.to(device)  # crop borders (loss)
                    offset = (3 * patch_size - 128) // 2

                self.optimizer.zero_grad()  # zero the parameter gradients
                lrs = lrs.float().to(device)
                hr_maps = hr_maps.float().to(device)
                hrs = hrs.float().to(device)

                srs = self.fus_model(lrs)  # fuse multi frames (B, 1, 3*W, 3*H)

                hrs_cropped = hrs[:, offset:(offset + 128), offset:(offset + 128)].view(-1, 1, 128, 128)
                sr_cropped = srs[:, :, offset:(offset + 128), offset:(offset + 128)]
                shifts = self.reg_model(torch.cat([hrs_cropped, sr_cropped], 1))
                batch_size, seq_len, height, width = srs.shape
                srs = srs.view(-1, 1, height, width)
                srs_shifted = RegistrationModel.transform(shifts, srs)
                srs_shifted = srs_shifted.view(-1, seq_len, height, width)[:, 0]
                cropped_mask = torch_mask[0] * hr_maps  # Compute current mask (Batch size, W, H)
                # loss
                loss = -self._get_loss(srs_shifted, hrs, cropped_mask)
                loss = torch.mean(loss)
                loss += config["training"]["lambda"] * torch.mean(shifts) ** 2
                # back-propagation
                loss.backward()
                self.optimizer.step()
                epoch_loss = loss.detach().cpu().numpy() * len(hrs) / len(self.dataloaders['train'].dataset)
                train_loss += epoch_loss

            # Eval
            self.fus_model.eval()
            val_score = 0.0  # monitor val score

            for lrs, hrs, hr_maps, names in self.dataloaders['val']:
                lrs = lrs.float().to(device)
                hrs = hrs.numpy()
                hr_maps = hr_maps.numpy()

                srs = self.fus_model(lrs)[:, 0]  # fuse multi frames (B, 1, 3*W, 3*H)

                # compute ESA score
                srs = srs.detach().cpu().numpy()
                for i in range(srs.shape[0]):  # batch size
                    if self.baseline_cpsnrs is None:
                        val_score -= shift_cpsnr(np.clip(srs[i], 0, 1), hrs[i], hr_maps[i])
                    else:
                        ESA = self.baseline_cpsnrs[names[i]]
                        val_score += ESA / shift_cpsnr(np.clip(srs[i], 0, 1), hrs[i], hr_maps[i])

            val_score /= len(self.dataloaders['val'].dataset)

            if best_score > val_score:
                torch.save(self.fus_model.state_dict(), os.path.join(checkpoint_dir_run, FUS_MODEL))
                torch.save(self.reg_model.state_dict(), os.path.join(checkpoint_dir_run, REG_MODEL))
                best_score = val_score

            writer.add_image('SR Image', (srs[0] - np.min(srs[0])) / np.max(srs[0]), epoch, dataformats='HW')
            error_map = hrs[0] - srs[0]
            writer.add_image('Error Map', error_map, epoch, dataformats='HW')
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/val_loss", val_score, epoch)
            scheduler.step(val_score)
        writer.close()

    def load_checkpoint(self, checkpoint):
        """
        load model weights from checkpoint
        """
        self.fus_model = FusionModel().to(self.device)
        self.fus_model.load_state_dict(torch.load(checkpoint))

    def evaluate(self, train_dataset, val_dataset, test_dataset, baseline_cpsnrs):
        """
        Benchmark train, validation, test dataset against baseline cPNSRs
        """
        self.fus_model.eval()
        seq_len = self.config['training']['seq_len']
        scores = {}
        clearance = {}
        part = {}
        for s, imgset_dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            for imgset in tqdm(imgset_dataset):
                sr, sc_pnsr = get_sr_and_score(imgset, self.fus_model, seq_len=seq_len)
                scores[imgset['name']] = sc_pnsr
                clearance[imgset['name']] = imgset['clearances']
                part[imgset['name']] = s
        results = benchmark(baseline_cpsnrs, scores, part, clearance)
        return results

    def generate_submission(self, checkpoint, out, seq_len=8):
        """
        Generate submission file
        """
        self.load_checkpoint(checkpoint)

        test_set_directories = get_imgset_dirs(os.path.join(self.config['folder']['data_dir'], "test"))
        test_dataset = ImgDataset(imgset_dir=test_set_directories, seq_len=seq_len)

        for imgset in tqdm(test_dataset):
            imgset_name = imgset['name']

            if imgset.__class__ is ImgSet:
                collator = CollateFn(seq_len=seq_len)
                lrs, hrs, hr_maps, names = collator([imgset])
            else:
                raise Exception("please check imgset type")

            lrs = lrs.float().to(self.device)

            sr = self.fus_model(lrs)[:, 0]
            sr = sr.detach().cpu().numpy()[0]

            sr = img_as_uint(sr)

            os.makedirs(out, exist_ok=True)

            # normalize and safe resulting image in temporary folder (complains on low contrast if not suppressed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(os.path.join(out, imgset_name + '.png'), sr)

        sub_archive = out + '/submission.zip'  # name of submission archive
        zf = ZipFile(sub_archive, mode='w')
        try:
            for img in os.listdir(out):
                if not img.startswith('imgset'):  # ignore the .zip-file itself
                    continue
                zf.write(os.path.join(out, img), arcname=img)
        finally:
            zf.close()

    def _get_loss(self, srs, hrs, hr_maps):
        """
        Compute loss as cPSNR: https://kelvins.esa.int/proba-v-super-resolution/scoring/
        """
        criterion = nn.MSELoss(reduction='none')
        nclear = torch.sum(hr_maps, dim=(1, 2))  # Number of clear pixels in target image
        bright = torch.sum(hr_maps * (hrs - srs), dim=(1, 2)).clone().detach() / nclear  # Correct for brightness
        loss = torch.sum(hr_maps * criterion(srs + bright.view(-1, 1, 1), hrs),
                         dim=(1, 2)) / nclear  # cMSE(A,B) for each point
        return -10 * torch.log10(loss)  # cPSNR
