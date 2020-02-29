import pathlib
import platform
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from utils.utils import init_fn
from datasets.music import midi_to_roll

class Dataloaders:
    def __init__(self, hp, args):
        pathlist = list(pathlib.Path(hp.data.path).glob('**/*.[Mm][Ii][Dd]'))
        np.random.shuffle(pathlist)
        self.split = hp.data.valid_split
        self.trainlist = pathlist[:-self.split * 2]
        self.validlist = pathlist[-self.split * 2:-self.split]
        self.testlist = pathlist[-self.split:]
        self.hp = hp
        self.args = args

    def get_pathlist(self, mode):
        if mode == "train":
            return self.trainlist
        if mode == "val":
            return self.validlist
        if mode == "test":
            return self.testlist
        raise NotImplementedError

    def get_dataloader(self, mode):
        if self.hp.data.dataset == "synthetic":
            dataset = CopyDataSet(self.hp, self.args)
        elif self.hp.data.dataset == "music":
            dataset = MusicDataset(self.hp, self.args, self.get_pathlist(mode), mode == "train")
        else:
            raise NotImplementedError
        sampler = RandomSampler(dataset) if platform.system() == "Windows"\
                else DistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=init_fn,
            sampler=sampler
        )

class CopyDataSet(Dataset):
    def __init__(self, hp, args):
        super(CopyDataSet, self).__init__()
        self.dataset_length = hp.data.dataset_length
        self.data_length = hp.data.data_length
        self.padding_idx = -1
        self.vocab = hp.data.vocab

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        src = torch.from_numpy(
            np.random.randint(1, self.vocab, size=(self.data_length // 2), dtype=np.int64)
        )
        src[0] = 0
        src = torch.cat([src, src, torch.LongTensor([0])], dim=0)
        x = src[:-1]
        y = src[1:]
        mask = y != self.padding_idx
        return x, y, mask

class MusicDataset(Dataset):
    def __init__(self, hp, args, pathlist, augment):
        super(MusicDataset, self).__init__()
        self.data_length = hp.data.data_length
        self.padding_idx = hp.data.vocab[0] - 1
        self.pathlist = pathlist
        self.augment = augment

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        src = torch.from_numpy(
            midi_to_roll(self.pathlist[idx], self.data_length + 1, self.augment)
        )
        mask = src[1:, 0] != self.padding_idx
        return src[:-1], src[1:], mask
