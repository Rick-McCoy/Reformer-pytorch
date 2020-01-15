import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset

def create_dataloader(hp, args, mode):
    if hp.data.name == "synthetic":
        return DataLoader(
            CopyDataSet(hp, args, mode),
            batch_size=args.batch_size,
            shuffle=(mode == "train"),
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        raise NotImplementedError

class CopyDataSet(Dataset):
    def __init__(self, hp, args, mode):
        super(CopyDataSet).__init__()
        self.dataset_length = hp.data.dataset_length
        self.max_data_length = hp.data.max_data_length
        self.data_length = hp.data.data_length
        self.vocab = hp.data.vocab
        self.mode = mode

    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        src = torch.from_numpy(np.random.randint(1, self.vocab, size=(self.max_data_length // 2))).long()
        src[0] = 0
        src = torch.cat([src, src, torch.LongTensor([0])], dim=0)
        x = src[:-1]
        y = src[1:]
        mask = x != 1e9
        return x, y, mask
