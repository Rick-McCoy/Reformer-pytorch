import torch

import numpy as np

def init_fn(worker_id):
    return np.random.seed(torch.initial_seed() % (2 ** 32) + worker_id)

def merge_hp(hp, args):
    for key, value in hp.model.items():
        setattr(args, key, value)
    for key, value in hp.data.items():
        setattr(args, key, value)
    for key, value in hp.train.items():
        setattr(args, key, value)
    return args

def deterministic_dropout(x: torch.Tensor, seed=0, dropout=0):
    generator = torch.Generator(device=x.get_device())
    generator.manual_seed(seed)
    dropout_mask = torch.bernoulli(x, p=1 - dropout, generator=generator)
    return dropout_mask * x / (1 - dropout)
