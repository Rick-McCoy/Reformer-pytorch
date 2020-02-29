'''
Implements LSH Attention in MultiHeadLSHAttention
'''
import math
import torch
import torch.nn.functional as F

from torch import nn
from utils.utils import deterministic_dropout, look_back, reverse_sort,\
                        expand, get_dup_keys, expand_gather

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
        length = inp.size(1)
        inp = F.normalize(inp, p=2, dim=-1)
        # [batch * head, length, d_k]
        if random:
            self.rand_matrix = torch.randn(
                [batch_size, self.d_k, self.rounds, n_buckets // 2],
                device=inp.get_device()
            )
            # [batch * head, d_k, rounds, n_buckets // 2]
            self.rand_matrix /= torch.norm(self.rand_matrix, dim=1, keepdim=True)
            # [batch * head, d_k, rounds, n_buckets // 2]
        matmul = torch.einsum('...ij,...jkl->...ikl', inp, self.rand_matrix)
        # [batch * head, length, rounds, n_buckets // 2]
        hashes = torch.argmax(torch.cat([matmul, -matmul], dim=-1), dim=-1).int()
        # [batch * head, length, rounds]
        arange = hashes.new_empty((1, length, 1))
        # [1, length, 1]
        hashes = hashes * length + torch.arange(length, out=arange).expand_as(hashes)
        # [batch * head, length, rounds]
        return hashes

class LSHAttention(nn.Module):
    '''
    Implements LSHAttention
    class is used to save LocalitySensitiveHash
    '''
    def __init__(self, hp, args):
        super(LSHAttention, self).__init__()
        self.d_k = hp.model.d_model // hp.model.head
        self.rounds = hp.model.rounds
        self.dropout = hp.model.dropout
        self.bucket_length = hp.model.bucket_length
        self.lsh = LocalitySensitiveHash(hp, args)

    def forward(self, query, value, seed, random=True):
        length = query.size(1)
        n_buckets = length // self.bucket_length

        sorted_hashes, hash_indice = torch.sort(self.lsh(query, n_buckets, random), dim=1)
        # [batch * head, length, rounds]
        original_indice = reverse_sort(hash_indice, dim=1)
        # [batch * head, length, rounds]

        reordered_query = expand_gather(
            expand(query, dim=3, num=self.rounds), dim=1,\
            index=hash_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        reordered_query = reordered_query.reshape(
            -1, n_buckets, self.bucket_length, self.d_k, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, d_k, rounds]
        lookback_key = F.normalize(look_back(reordered_query), p=2, dim=-2)
        # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
        matmul_qk = torch.einsum(
            '...ijk,...ljk->...ilk', reordered_query, lookback_key
        ) / math.sqrt(self.d_k)
        # [batch * head, n_buckets, bucket_length, bucket_length * 2, rounds]

        sorted_hashes = sorted_hashes.reshape(
            -1, n_buckets, self.bucket_length, self.rounds
        ) // length
        # [batch * head, n_buckets, bucket_length, rounds]
        matmul_qk.masked_fill_(
            mask=(sorted_hashes[..., None, :] != look_back(sorted_hashes)[..., None, :, :]),\
            value=-1e9
        )

        query_indice = hash_indice.reshape(
            -1, n_buckets, self.bucket_length, self.rounds
        ).int()
        # [batch * head, n_buckets, bucket_length, rounds]
        key_indice = look_back(query_indice)
        # [batch * head, n_buckets, bucket_length * 2, rounds]
        matmul_qk.masked_fill_(
            mask=(query_indice[..., None, :] < key_indice[..., None, :, :]), value=-1e9
        )
        matmul_qk.masked_fill_(
            mask=(query_indice[..., None, :] == key_indice[..., None, :, :]), value=-1e5
        )

        key_indice = expand(key_indice, dim=2, num=self.bucket_length).flatten(1, 2)
        # [batch * head, length, bucket_length * 2, rounds]
        key_indice = expand_gather(
            key_indice,
            dim=1, index=original_indice,
            expand_dim=2, num=self.bucket_length * 2
        )
        # [batch * head, length, bucket_length * 2, rounds]
        count_key = get_dup_keys(
            key_indice.flatten(-2, -1), self.rounds
        ).reshape(-1, length, self.bucket_length * 2, self.rounds)
        # [batch * head, length, bucket_length * 2, rounds]
        count_key = expand_gather(
            count_key, dim=1, index=hash_indice, expand_dim=2, num=self.bucket_length * 2
        )
        # [batch * head, length, bucket_length * 2, rounds]
        matmul_qk = matmul_qk.flatten(1, 2)
        # [batch * head, length, bucket_length * 2, rounds]
        logsumexp_qk = torch.logsumexp(matmul_qk, dim=2)
        # [batch * head, length, rounds]
        softmax_qk = torch.exp(matmul_qk - count_key.float().log_() - logsumexp_qk[..., None, :])
        # [batch * head, length, bucket_length * 2, rounds]

        if self.training:
            softmax_qk = deterministic_dropout(softmax_qk, seed=seed, dropout=self.dropout)
            # [batch * head, length, bucket_length * 2, rounds]

        reordered_value = expand_gather(
            expand(value, dim=3, num=self.rounds), dim=1,\
            index=hash_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        reordered_value = reordered_value.reshape(
            -1, n_buckets, self.bucket_length, self.d_k, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, d_k, rounds]

        softmax_qk = softmax_qk.reshape(
            -1, n_buckets, self.bucket_length, self.bucket_length * 2, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, bucket_length * 2, rounds]

        attention = torch.einsum('...ijl,...jkl->...ikl', softmax_qk, look_back(reordered_value))
        # [batch * head, n_buckets, bucket_length, d_k, rounds]
        attention = attention.flatten(1, 2)
        # [batch * head, length, d_k, rounds]
        attention = expand_gather(
            attention, dim=1, index=original_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        logsumexp_qk = torch.gather(logsumexp_qk, dim=1, index=original_indice)
        # [batch * head, length, rounds]
        logsumexp_qk = F.softmax(logsumexp_qk, dim=1)
        # [batch * head, length, rounds]
        attention = torch.einsum('...ij,...j->...i', attention, logsumexp_qk)
        # [batch * head, length, d_k]

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
        self.chunk = hp.model.chunk
        self.linear_query = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_value = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_out = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.lshattention = LSHAttention(hp, args)

    def forward(self, input_tensor, seed, random=True):
        length = input_tensor.size(1)

        query = self.linear_query(input_tensor).reshape(-1, length, self.head, self.d_k).transpose_(1, 2)
        # [batch, head, length, d_k]
        value = self.linear_value(input_tensor).reshape(-1, length, self.head, self.d_k).transpose_(1, 2)
        # [batch, head, length, d_k]

        chunked_query = torch.chunk(query.flatten(0, 1), chunks=self.chunk, dim=0)
        # [batch * head // chunk, length, d_k]
        chunked_value = torch.chunk(value.flatten(0, 1), chunks=self.chunk, dim=0)
        # [batch * head // chunk, length, d_k]

        attention = torch.cat([
            self.lshattention(q, v, seed + i, random) for q, v, i\
                in zip(chunked_query, chunked_value, range(self.chunk))
        ], dim=0).reshape(-1, self.head, length, self.d_k)
        # [batch, head, length, d_k]

        attention = attention.transpose(1, 2).flatten(-2, -1)
        # [batch, length, d_model]

        return self.linear_out(attention)
