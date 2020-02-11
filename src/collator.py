import torch


class CollateFn:
    """
    Collate lrs, hr, hr_map, and name and convert them torch ready for processing
    """

    def __init__(self, seq_len=8):
        self.seq_len = seq_len

    def __call__(self, batch):
        return self.collate_fn(batch)

    def collate_fn(self, batch):
        lrs_batch = []  # batch of low-res images
        hr_batch = []  # batch of high-res image
        hm_batch = []  # batch of high-res clearance map
        name_batch = []  # batch of imgset names

        train_batch = True

        for imgset in batch:
            lrs = imgset['lrs']
            assert lrs.size(0) >= self.seq_len
            lrs_batch.append(lrs[:self.seq_len])
            hr = imgset['hr']
            if train_batch and hr is not None:
                hr_batch.append(hr)
            else:
                train_batch = False
            hm_batch.append(imgset['hr_map'])
            name_batch.append(imgset['name'])

        lrs_batch = torch.stack(lrs_batch, dim=0)

        if train_batch:
            hr_batch = torch.stack(hr_batch, dim=0)
            hm_batch = torch.stack(hm_batch, dim=0)

        return lrs_batch, hr_batch, hm_batch, name_batch
