import torch

import numpy as np

def init_fn(worker_id):
    return np.random.seed(torch.initial_seed() % (2 ** 32) + worker_id)
