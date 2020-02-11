import glob
import skimage
import torch
import os
import numpy as np

from skimage import io
from collections import OrderedDict
from os.path import join, exists, basename, isfile
from torch.utils.data import Dataset

CLEARANCE_SCORES = "clearance_score.npy"


def save_clearance_score(imgset_dir):
    """
    Re-use clearance score per imgset instead of re-computing it
    """
    indices = np.array([os.path.basename(path)[2:-4]
                        for path in glob.glob(os.path.join(imgset_dir, 'QM*.png'))])
    indices = np.sort(indices)
    lr_maps = np.array([io.imread(os.path.join(imgset_dir, f'QM{i}.png')) for i in indices], dtype=np.uint16)
    scores = lr_maps.sum(axis=(1, 2))
    np.save(os.path.join(imgset_dir, CLEARANCE_SCORES), scores)
    return scores


class ImgSet(OrderedDict):
    """
    Ordered dictionary for 'imgset'
    """

    def __init__(self, *args, **kwargs):
        super(ImgSet, self).__init__(*args, **kwargs)

    def __repr__(self):
        dict_info = f"{'name':>10} : {self['name']}"
        for name, v in self.items():
            if hasattr(v, 'shape'):
                dict_info += f"\n{name:>10} : {v.shape} {v.__class__.__name__} ({v.dtype})"
            else:
                dict_info += f"\n{name:>10} : {v.__class__.__name__} ({v})"
        return dict_info


def sample_clearest(clearances, n=None, beta=50.0, seed=0):
    """
    Sample according to clearance score with some degree of random effect.
    """
    if seed is not None:
        np.random.seed(seed)
    e_c = np.exp(beta * clearances / clearances.max())
    p = e_c / e_c.sum()
    idx = range(len(p))
    i_sample = np.random.choice(idx, size=n, p=p, replace=False)
    return i_sample


def read_imgset(imgset_dir, seq_len):
    """
    Read images in a given 'imgset' directory
    """
    indices = np.array([basename(path)[2:-4] for path in glob.glob(join(imgset_dir, 'QM*.png'))])
    indices = np.sort(indices)

    if isfile(join(imgset_dir, CLEARANCE_SCORES)):
        clearance = np.load(join(imgset_dir, CLEARANCE_SCORES))  # load clearance scores
    else:
        clearance = save_clearance_score(imgset_dir)

    seq_len = min(seq_len, len(indices))

    i_samples = sample_clearest(clearance, n=seq_len)
    indices = indices[i_samples]
    clearance = clearance[i_samples]

    lr_images = np.array([io.imread(join(imgset_dir, f'LR{i}.png')) for i in indices], dtype=np.uint16)

    hr_map = np.array(io.imread(join(imgset_dir, 'SM.png')), dtype=np.bool)
    if exists(join(imgset_dir, 'HR.png')):
        hr = np.array(io.imread(join(imgset_dir, 'HR.png')), dtype=np.uint16)
    else:
        hr = None

    imgset = ImgSet(name=basename(imgset_dir),
                    lrs=np.array(lr_images),
                    hr=hr,
                    hr_map=hr_map,
                    clearances=clearance,
                    )
    return imgset


class ImgDataset(Dataset):
    """
    Customized dataset for imgset dirs
    """

    def __init__(self, imgset_dir, seq_len=8):
        super().__init__()
        self.imgset_dir = imgset_dir
        self.imgset_names = {basename(im_dir): im_dir for im_dir in imgset_dir}
        self.seq_len = seq_len

    def __len__(self):
        return len(self.imgset_dir)

    def __getitem__(self, index):
        if isinstance(index, int):
            imgset_dir = self.imgset_dir[index]
        elif isinstance(index, str):
            imgset_dir = self.imgset_names[index]
        else:
            raise KeyError('index must be int or string')

        imgset = read_imgset(imgset_dir, self.seq_len)
        imgset['lrs'] = torch.from_numpy(skimage.img_as_float(imgset['lrs']).astype(np.float32))
        if imgset['hr'] is not None:
            imgset['hr'] = torch.from_numpy(skimage.img_as_float(imgset['hr']).astype(np.float32))
            imgset['hr_map'] = torch.from_numpy(imgset['hr_map'].astype(np.float32))

        return imgset
