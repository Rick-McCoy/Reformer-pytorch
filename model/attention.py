'''
Implements LSH Attention in MultiHeadLSHAttention
'''
import math
import torch
import torch.nn.functional as F

from torch import nn
from utils.utils import deterministic_dropout

def look_back(input_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Looks back one bucket
    '''
    shift = torch.cat([input_tensor[:, -1:], input_tensor[:, :-1]], dim=1)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
    concat = torch.cat([shift, input_tensor], dim=2)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]
    return concat

def reverse_sort(indice: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    Unsorts sorted indice
    '''
    size = indice.size(dim)
    new_size = [1] * dim + [size] + [1] * (indice.dim() - dim - 1)
    arange = indice.new_empty(size=new_size)
    torch.arange(size, out=arange)
    arange = arange.expand_as(indice)
    new_indice = torch.empty_like(indice)
    new_indice.scatter_(dim=dim, index=indice, src=arange)
    return new_indice

def expand(input_tensor: torch.Tensor, dim=0, num=1) -> torch.Tensor:
    '''
    Shortcut for unsqueeze + expand
    '''
    new_size = [-1] * dim + [num] + [-1] * (input_tensor.dim() - dim)
    return input_tensor.unsqueeze(dim=dim).expand(new_size)

def get_dup_keys(input_tensor: torch.Tensor, rounds=0) -> torch.Tensor:
    sorted_flat_key, flat_key_indice = torch.sort(input_tensor, dim=-1)
    # [batch * head, length, bucket_length * 4 * rounds]
    count_shift_keys = torch.ones_like(sorted_flat_key).int()
    # [batch * head, length, bucket_length * 4 * rounds]
    for i in range(1, rounds):
        equiv_flat_key = (sorted_flat_key[..., i:] == sorted_flat_key[..., :-i]).int()
        count_shift_keys[..., i:] += equiv_flat_key
        count_shift_keys[..., :-i] += equiv_flat_key
    count_key_indice = reverse_sort(flat_key_indice, dim=2)
    # [batch * head, length, bucket_length * 4 * rounds]
    return torch.gather(count_shift_keys, dim=-1, index=count_key_indice)


class LocalitySensitiveHash(nn.Module):
    '''
    Implements Locality Sensitive Hash
    class is used to save random matrix used for hashing
    '''
    def __init__(self, hp, args):
        super(LocalitySensitiveHash, self).__init__()
        self.d_k = hp.model.d_model // hp.model.head
        self.rounds = hp.model.rounds
        self.rand_matrix = None

    def forward(self, inp: torch.Tensor, n_buckets=0, random=True):
        batch_size = inp.size(0)
        if random:
            self.rand_matrix = torch.randn(
                [batch_size, self.d_k, self.rounds, n_buckets // 2],
                device=inp.get_device()
            )
            # [batch * head, d_k, rounds, n_buckets // 2]
            self.rand_matrix /= torch.norm(self.rand_matrix, dim=1, keepdim=True)
            # [batch * head, d_k, rounds, n_buckets // 2]
        hashes = torch.einsum('...ij,...jkl->...ikl', inp, self.rand_matrix)
        # [batch * head, length, rounds, n_buckets // 2]
        return torch.argmax(torch.cat([hashes, -hashes], dim=-1), dim=-1).int()

class LSHAttention(nn.Module):
    '''
    Implements LSHAttention
    class is used to save LocalitySensitiveHash
    '''
    def __init__(self, hp, args):
        super(LSHAttention, self).__init__()
        self.head = hp.model.head
        self.d_k = hp.model.d_model // hp.model.head
        self.rounds = hp.model.rounds
        self.dropout = hp.model.dropout
        self.bucket_length = hp.model.bucket_length
        self.lsh = LocalitySensitiveHash(hp, args)

    def forward(self, query, value, seed, random=True):
        length = query.size(2)
        n_buckets = length // self.bucket_length

        normalized_query = query / torch.norm(query, dim=-1, keepdim=True)
        # [batch, head, length, d_k]
        flattened_query = normalized_query.flatten(0, 1)
        # [batch * head, length, d_k]

        hashes = self.lsh(flattened_query, n_buckets, random)
        # [batch * head, length, rounds]
        sorted_hashes, hash_indice = torch.sort(hashes, dim=1)
        # [batch * head, length, rounds]
        expanded_hash_indice = expand(hash_indice, dim=2, num=self.d_k)
        # [batch * head, length, d_k, rounds]

        expanded_query = expand(flattened_query, dim=3, num=self.rounds)
        # [batch * head, length, d_k, rounds]
        reordered_query = torch.gather(expanded_query, dim=1, index=expanded_hash_indice)
        # [batch * head, length, d_k, rounds]
        reordered_query = reordered_query.reshape(
            -1, n_buckets // 2, self.bucket_length * 2, self.d_k, self.rounds
        )
        # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]

        lookback_key = look_back(reordered_query)
        # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]

        scores = torch.einsum(
            '...ijk,...ljk->...ilk', reordered_query, lookback_key
        ) / math.sqrt(self.d_k)
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]

        sorted_hashes = sorted_hashes.reshape(
            -1, n_buckets // 2, self.bucket_length * 2, self.rounds
        )
        # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
        lookback_hash = look_back(sorted_hashes)
        # [batch * head, n_buckets // 2, bucket_length * 4, rounds]
        hash_equiv_mask = sorted_hashes[..., None, :] != lookback_hash[..., None, :, :]
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        scores.masked_fill_(mask=hash_equiv_mask, value=-1e9)

        query_indice = hash_indice.reshape(
            -1, n_buckets // 2, self.bucket_length * 2, self.rounds
        )
        # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
        key_indice = look_back(query_indice)
        # [batch * head, n_buckets // 2, bucket_length * 4, rounds]

        causal_mask = query_indice[..., None, :] < key_indice[..., None, :, :]
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        scores.masked_fill_(mask=causal_mask, value=-1e9)

        indice_equiv_mask = query_indice[..., None, :] == key_indice[..., None, :, :]
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        scores.masked_fill_(mask=indice_equiv_mask, value=-1e5)

        original_indice = reverse_sort(hash_indice, dim=1)
        # [batch * head, length, rounds]
        score_indice = expand(original_indice, dim=2, num=self.bucket_length * 4)
        # [batch * head, length, bucket_length * 4, rounds]

        expanded_key_indice = expand(key_indice, dim=2, num=self.bucket_length * 2)
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        reordered_key_indice = torch.gather(
            expanded_key_indice.flatten(1, 2), dim=1, index=score_indice
        )
        # [batch * head, length, bucket_length * 4, rounds]
        flat_reordered_key = reordered_key_indice.flatten(-2, -1)
        # [batch * head, length, bucket_length * 4 * rounds]
        count_key = get_dup_keys(flat_reordered_key.int(), self.rounds)
        # [batch * head, length, bucket_length * 4 * rounds]
        scores = scores.flatten(1, 2)
        # [batch * head, length, bucket_length * 4, rounds]
        scores = torch.gather(scores, dim=1, index=score_indice)
        # [batch * head, length, bucket_length * 4, rounds]
        scores = scores.flatten(-2, -1) - count_key.float().log().detach()
        # [batch * head, length, bucket_length * 4 * rounds]
        p_attn = F.softmax(scores, dim=-1)
        # [batch * head, length, bucket_length * 4 * rounds]

        if self.training:
            p_attn = deterministic_dropout(p_attn, seed=seed, dropout=self.dropout)
            # [batch * head, length, bucket_length * 4 * rounds]

        p_attn = p_attn.reshape(-1, length, self.bucket_length * 4, self.rounds)
        # [batch * head, length, bucket_length * 4, rounds]

        flattened_value = expand(value.flatten(0, 1), dim=3, num=self.rounds)
        # [batch * head, length, d_k, rounds]
        reordered_value = torch.gather(flattened_value, dim=1, index=expanded_hash_indice)
        # [batch * head, length, d_k, rounds]
        reshaped_value = reordered_value.reshape(
            -1, n_buckets // 2, self.bucket_length * 2, self.d_k, self.rounds
        )
        # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
        lookback_value = look_back(reshaped_value)
        # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]

        attn_indice = expand(hash_indice, dim=2, num=self.bucket_length * 4)
        # [batch * head, length, bucket_length * 4, rounds]
        reordered_p_attn = torch.gather(p_attn, dim=1, index=attn_indice)
        # [batch * head, length, bucket_length * 4, rounds]
        new_p_attn = reordered_p_attn.reshape(
            -1, n_buckets // 2, self.bucket_length * 2, self.bucket_length * 4, self.rounds
        )
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]

        attention = torch.einsum('...ijl,...jkl->...ikl', new_p_attn, lookback_value)
        # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
        attention = attention.flatten(1, 2)
        # [batch * head, length, d_k, rounds]
        new_indice = expand(original_indice, dim=2, num=self.d_k)
        # [batch * head, length, d_k, rounds]
        attention = torch.gather(attention, dim=1, index=new_indice).sum(dim=-1)
        # [batch * head, length, d_k]
        attention = attention.reshape(-1, self.head, length, self.d_k)
        # [batch, head, length, d_k]

        return attention

class MultiRoundLSHAttention(nn.Module):
    '''
    Implements Multi Round LSH Attention
    class is defined to save LSHAttention
    '''
    def __init__(self, hp, args):
        super(MultiRoundLSHAttention, self).__init__()
        self.d_k = hp.model.d_model // hp.model.head
        self.head = hp.model.head
        self.linear_query = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_value = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_out = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.lshattention = LSHAttention(hp, args)

    def forward(self, query, value, seed, random=True):
        length = query.size(1)

        query = self.linear_query(query).reshape(-1, length, self.head, self.d_k).transpose(1, 2)
        # [batch, head, length, d_k]
        value = self.linear_value(value).reshape(-1, length, self.head, self.d_k).transpose(1, 2)
        # [batch, head, length, d_k]

        attention = self.lshattention(query, value, seed, random)
        # [batch, head, length, d_k]

        attention = attention.transpose(1, 2).flatten(-2, -1)
        # [batch, length, d_model]

        return self.linear_out(attention)
