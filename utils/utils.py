import os
import time
import torch

import numpy as np

from tqdm import tqdm
from datasets.music import roll_to_midi

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

def look_back(input_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Looks back one bucket
    '''
    shift = torch.cat([input_tensor[:, -1:], input_tensor[:, :-1]], dim=1)
    # [batch * head, n_buckets, bucket_length, d_k, rounds]
    concat = torch.cat([shift, input_tensor], dim=2)
    # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
    return concat

def reverse_sort(indice: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    Unsorts sorted indice
    '''
    new_size = [1] * indice.dim()
    new_size[dim] = indice.size(dim)
    arange = indice.new_empty(size=new_size)
    torch.arange(new_size[dim], out=arange)
    arange = arange.expand_as(indice)
    new_indice = torch.empty_like(indice)
    new_indice.scatter_(dim=dim, index=indice, src=arange)
    return new_indice

def expand(input_tensor: torch.Tensor, dim=0, num=1) -> torch.Tensor:
    '''
    Shortcut for unsqueeze + expand
    '''
    new_size = [-1] * (input_tensor.dim() + 1)
    new_size[dim] = num
    return input_tensor.unsqueeze(dim=dim).expand(new_size)

def expand_gather(input_tensor: torch.Tensor, dim: int, index: torch.Tensor, expand_dim=0, num=1) -> torch.Tensor:
    expanded_index = expand(index, dim=expand_dim, num=num)
    return input_tensor.gather(dim=dim, index=expanded_index)

def get_dup_keys(input_tensor: torch.Tensor, rounds=0) -> torch.Tensor:
    sorted_flat_key, flat_key_indice = torch.sort(input_tensor, dim=-1)
    # [batch * head, length, bucket_length * 2 * rounds]
    count_shift_keys = torch.ones_like(sorted_flat_key)
    # [batch * head, length, bucket_length * 2 * rounds]
    for i in range(1, rounds):
        equiv_flat_key = (sorted_flat_key[..., i:] == sorted_flat_key[..., :-i]).int()
        count_shift_keys[..., i:] += equiv_flat_key
        count_shift_keys[..., :-i] += equiv_flat_key
    count_key_indice = reverse_sort(flat_key_indice, dim=2)
    # [batch * head, length, bucket_length * 2 * rounds]
    return torch.gather(count_shift_keys, dim=-1, index=count_key_indice)

def top_p_sample(prob: torch.Tensor, perc=0.5) -> np.array:
    sorted_prob, sorted_indices = torch.sort(prob, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_prob, dim=-1)
    mask = cumsum < perc
    one_more_indice = mask.long().sum(dim=-1, keepdim=True)
    mask.scatter_(dim=-1, index=one_more_indice, value=True)
    sorted_prob.masked_fill_(~mask, value=0.0)
    masked_prob = sorted_prob.gather(dim=-1, index=reverse_sort(sorted_indices, dim=-1))
    return torch.multinomial(masked_prob, num_samples=1)

def save_to_midi(sample: np.array):
    try:
        song = roll_to_midi(sample)
        name = str(int(time.time())) + '.mid'
        song.write(os.path.join('samples', name))
        tqdm.write('Saved to ' + os.path.join('samples', name))
    except AssertionError as error:
        tqdm.write(str(error))
        tqdm.write('Failed to generate sample')
