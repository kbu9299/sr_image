import csv
import os
import torch
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from collator import CollateFn
from sklearn.model_selection import train_test_split
from dataloader import ImgSet, ImgDataset


def set_random_seed(seed=0):
    """
    For reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_crop_mask(patch_size, crop_size):
    """
    Computes a mask to crop borders.
    """
    mask = np.ones((1, 1, 3 * patch_size, 3 * patch_size))  # crop_mask for loss (B, C, W, H)
    mask[0, 0, :crop_size, :] = 0
    mask[0, 0, -crop_size:, :] = 0
    mask[0, 0, :, :crop_size] = 0
    mask[0, 0, :, -crop_size:] = 0
    torch_mask = torch.from_numpy(mask).type(torch.FloatTensor)
    return torch_mask


def load_baseline_cpsnr(path):
    """
    Reads the baseline cPSNR scores from `path`.
    """
    scores = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            scores[row[0].strip()] = float(row[1].strip())
    return scores


def get_patch(img, x, y, size=32):
    patch = img[..., x:(x + size), y:(y + size)]  # using ellipsis to slice arbitrary ndarrays
    return patch


def get_cpsnr(sr, hr, hr_map):
    """
    Clear Peak Signal-to_Noise Ratio.
    """
    if len(sr.shape) == 2:
        sr = sr[None,]
        hr = hr[None,]
        hr_map = hr_map[None,]

    if sr.dtype.type is np.uint16:  # integer array is in the range [0, 65536]
        sr = sr / np.iinfo(np.uint16).max  # normalize in the range [0, 1]
    else:
        assert 0 <= sr.min() and sr.max() <= 1, 'sr.dtype must be either uint16 (range 0-65536) or float64 in (0, 1).'
    if hr.dtype.type is np.uint16:
        hr = hr / np.iinfo(np.uint16).max

    n_clear = np.sum(hr_map, axis=(1, 2))  # number of clear pixels in the high-res patch
    diff = hr - sr
    bias = np.sum(diff * hr_map, axis=(1, 2)) / n_clear  # brightness bias
    cmse = np.sum(np.square((diff - bias[:, None, None]) * hr_map), axis=(1, 2)) / n_clear
    cpsnr = -10 * np.log10(cmse)  # + 1e-10)

    if cpsnr.shape[0] == 1:
        cpsnr = cpsnr[0]

    return cpsnr


def patch_iterator(img, positions, size):
    """
    Iterator across patches of 'img' located in 'position'.
    """
    for x, y in positions:
        yield get_patch(img=img, x=x, y=y, size=size)


def shift_cpsnr(sr, hr, hr_map, border_w=3):
    """
    Computes the max CPSNR score across shifts of up to 'border_w' pixels
    """
    size = sr.shape[1] - (2 * border_w)  # patch size
    sr = get_patch(img=sr, x=border_w, y=border_w, size=size)
    pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
    iter_hr = patch_iterator(img=hr, positions=pos, size=size)
    iter_hr_map = patch_iterator(img=hr_map, positions=pos, size=size)
    site_cpsnr = np.array([get_cpsnr(sr, hr, hr_map)
                           for hr, hr_map in tqdm(zip(iter_hr, iter_hr_map), disable=(len(sr.shape) == 2))])
    max_cpsnr = np.max(site_cpsnr, axis=0)
    return max_cpsnr


def get_imgset_dirs(data_dir):
    """
    Returns a list of paths to directories, one for every imaset in `data_dir`.
    """
    imgset_dirs = []
    for channel_dir in ['RED', 'NIR']:
        path = os.path.join(data_dir, channel_dir)
        for imgset_name in os.listdir(path):
            imgset_dirs.append(os.path.join(path, imgset_name))
    return imgset_dirs


def get_sr_and_score(imgset, model, seq_len):
    """
    Super-resolved image and compute the score against High-resolution image (ground truth)
    """
    if imgset.__class__ is ImgSet:
        collator = CollateFn(seq_len=seq_len)
        lrs, hrs, hr_maps, names = collator([imgset])
    elif isinstance(imgset, tuple):  # imset is a tuple of batches
        lrs, hrs, hr_maps, names = imgset
    else:
        raise Exception("Please check imgset type")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lrs = lrs.float().to(device)

    sr = model(lrs)[:, 0]
    sr = sr.detach().cpu().numpy()[0]

    if len(hrs) > 0:
        sc_psnr = shift_cpsnr(sr=np.clip(sr, 0, 1), hr=hrs.numpy()[0], hr_map=hr_maps.numpy()[0])
    else:
        sc_psnr = None

    return sr, sc_psnr


def benchmark(baseline_cpsnrs, scores, part, clearance_scores):
    """
    Benchmark scores against ESA baseline
    """
    results = pd.DataFrame({'ESA': baseline_cpsnrs, 'model': scores, 'clr': clearance_scores, 'part': part, })
    results['score'] = results['ESA'] / results['model']
    results['mean_clr'] = results['clr'].map(np.mean)
    results['std_clr'] = results['clr'].map(np.std)

    return results


def load_data(config, seq_len, val_proportion=0.10):
    """
    Loads all the data for the ESA Kelvin competition (train, val, test, baseline)
    """
    data_directory = config["folder"]["data_dir"]
    baseline_cpsnrs = load_baseline_cpsnr(os.path.join(data_directory, "norm.csv"))
    train_set_directories = get_imgset_dirs(os.path.join(data_directory, "train"))
    test_set_directories = get_imgset_dirs(os.path.join(data_directory, "test"))
    train_list, val_list = train_test_split(train_set_directories, test_size=val_proportion, random_state=1,
                                            shuffle=True)
    train_dataset = ImgDataset(imgset_dir=train_list, seq_len=seq_len)
    val_dataset = ImgDataset(imgset_dir=val_list, seq_len=seq_len)
    test_dataset = ImgDataset(imgset_dir=test_set_directories, seq_len=seq_len)

    return train_dataset, val_dataset, test_dataset, baseline_cpsnrs
