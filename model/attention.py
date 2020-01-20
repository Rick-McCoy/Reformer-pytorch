import math
import torch
import torch.nn.functional as F

from torch import nn

def look_back(x) -> torch.Tensor:
    shift = torch.cat([x[:, -1:], x[:, :-1]], dim=1)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
    concat = torch.cat([shift, x], dim=2)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]
    return concat

def reverse_sort(x: torch.Tensor, dim: int) -> torch.Tensor:
    new_indices = torch.empty_like(x)
    new_size = [1] * dim + [x.size(dim)] + [1] * (x.ndimension() - dim - 1)
    arange = torch._dim_arange(x, dim=dim).reshape(new_size).expand_as(x)
    new_indices.scatter_(dim=dim, index=x, src=arange)
    return new_indices

class LocalitySensitiveHash(nn.Module):
    def __init__(self, hp, args):
        super(LocalitySensitiveHash, self).__init__()
        self.d_k = hp.model.d_model // hp.model.head
        self.rounds = hp.model.rounds
        self.n_buckets = hp.model.n_buckets

    def forward(self, inp: torch.Tensor, random=True):
        batch_size = inp.size(0)
        if random:
            self.rand_matrix = torch.randn([batch_size, self.d_k, self.rounds, self.n_buckets // 2], device=inp.get_device())
            # [batch * head, d_k, rounds, n_buckets // 2]
            self.rand_matrix /= torch.norm(self.rand_matrix, dim=1, keepdim=True)
            # [batch * head, d_k, rounds, n_buckets // 2]
        hashes = torch.einsum('...ij,...jkl->...ikl', inp, self.rand_matrix)
        # [batch * head, length, rounds, n_buckets // 2]
        return torch.argmax(torch.cat([hashes, -hashes], dim=-1), dim=-1)

class LSHAttention(nn.Module):
    def __init__(self, hp, args):
        super(LSHAttention, self).__init__()
        self.head = hp.model.head
        self.d_k = hp.model.d_model // hp.model.head
        self.rounds = hp.model.rounds
        self.dropout = hp.model.dropout
        self.n_buckets = hp.model.n_buckets
        self.lsh = LocalitySensitiveHash(hp, args)

    def forward(self, query: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, seed: int, random=True):
        length = query.size(2)
        bucket_length = length // self.n_buckets

        query = query / torch.norm(query, dim=-1, keepdim=True)
        # [batch, head, length, d_k]
        flattened_query = query.flatten(0, 1)
        # [batch * head, length, d_k]

        hashes = self.lsh(flattened_query, random)
        # [batch * head, length, rounds]
        sorted_hashes, hash_indices = torch.sort(hashes, dim=1)
        # [batch * head, length, rounds]
        expanded_hash_indices = hash_indices[:, :, None, :].expand(-1, -1, self.d_k, -1)
        # [batch * head, length, d_k, rounds]

        expanded_query = flattened_query[..., None].expand(-1, -1, -1, self.rounds)
        # [batch * head, length, d_k, rounds]
        reordered_query = torch.gather(expanded_query, dim=1, index=expanded_hash_indices)
        # [batch * head, length, d_k, rounds]
        reordered_query = reordered_query.reshape(-1, self.n_buckets // 2, bucket_length * 2, self.d_k, self.rounds)
        # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]

        lookback_key = look_back(reordered_query)
        # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]

        scores = torch.einsum('...ijk,...ljk->...ilk', reordered_query, lookback_key) / math.sqrt(self.d_k)
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]

        mask = mask[:, None, :, None].expand(-1, self.head, -1, self.rounds).flatten(0, 1)
        # [batch * head, length, rounds]
        reordered_mask = torch.gather(mask, dim=1, index=hash_indices)
        # [batch * head, length, rounds]
        reordered_mask = reordered_mask.reshape(-1, self.n_buckets // 2, bucket_length * 2, self.rounds)
        # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
        lookback_mask = look_back(reordered_mask)[..., None, :, :]
        # [batch * head, n_buckets // 2, 1, bucket_length * 4, rounds]
        scores.masked_fill_(mask=~lookback_mask, value=-1e9)

        sorted_hashes = sorted_hashes.reshape(-1, self.n_buckets // 2, bucket_length * 2, self.rounds)
        # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
        lookback_hash = look_back(sorted_hashes)
        # [batch * head, n_buckets // 2, bucket_length * 4, rounds]
        hash_equiv_mask = (sorted_hashes[..., None, :] != lookback_hash[..., None, :, :])
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        scores.masked_fill_(mask=hash_equiv_mask, value=-1e9)

        query_indices = hash_indices.reshape(-1, self.n_buckets // 2, bucket_length * 2, self.rounds)
        # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
        key_indices = look_back(query_indices)
        # [batch * head, n_buckets // 2, bucket_length * 4, rounds]

        causal_mask = query_indices[..., None, :] < key_indices[..., None, :, :]
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        scores.masked_fill_(mask=causal_mask, value=-1e9)

        indice_equiv_mask = query_indices[..., None, :] == key_indices[..., None, :, :]
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        scores.masked_fill_(mask=indice_equiv_mask, value=-1e5)

        original_indices = reverse_sort(hash_indices, dim=1)
        # [batch * head, length, rounds]
        score_indices = original_indices[..., None, :].expand(-1, -1, bucket_length * 4, -1)
        # [batch * head, length, bucket_length * 4, rounds]

        expanded_key_indices = key_indices[..., None, :, :].expand(-1, -1, bucket_length * 2, -1, -1)
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
        reordered_key_indices = torch.gather(expanded_key_indices.flatten(1, 2), dim=1, index=score_indices)
        # [batch * head, length, bucket_length * 4, rounds]
        flat_reordered_key = reordered_key_indices.flatten(-2, -1).flatten(0, 1)
        # [batch * head * length, bucket_length * 4 * rounds]
        sorted_flat_key, flat_key_indices = torch.sort(flat_reordered_key.int(), dim=-1)
        # [batch * head * length, bucket_length * 4 * rounds]
        count_shift_keys = torch.ones_like(sorted_flat_key).float()
        # [batch * head * length, bucket_length * 4 * rounds]
        for i in range(1, self.rounds):
            equiv_flat_key = (sorted_flat_key[..., i:] == sorted_flat_key[..., :-i]).float()
            count_shift_keys[..., i:] += equiv_flat_key
            count_shift_keys[..., :-i] += equiv_flat_key
        count_key_indices = reverse_sort(flat_key_indices, dim=1)
        # [batch * head * length, bucket_length * 4 * rounds]
        count_key = torch.gather(count_shift_keys, dim=-1, index=count_key_indices)
        # [batch * head * length, bucket_length * 4 * rounds]
        reshaped_count_key = count_key.reshape(-1, length, bucket_length * 4, self.rounds)
        # [batch * head, length, bucket_length * 4, rounds]
        scores = scores.flatten(1, 2)
        # [batch * head, length, bucket_length * 4, rounds]
        scores = torch.gather(scores, dim=1, index=score_indices)
        # [batch * head, length, bucket_length * 4, rounds]
        scores = scores - reshaped_count_key.log().detach()

        scores = scores.flatten(-2, -1)
        # [batch * head, length, bucket_length * 4 * rounds]
        p_attn = F.softmax(scores, dim=-1)
        # [batch * head, length, bucket_length * 4 * rounds]

        if self.training:
            generator = torch.Generator(device=p_attn.get_device())
            generator.manual_seed(seed)
            dropout_mask = torch.bernoulli(p_attn, p=1 - self.dropout, generator=generator)
            p_attn = dropout_mask * p_attn / (1 - self.dropout)

        p_attn = p_attn.reshape(-1, length, bucket_length * 4, self.rounds)
        # [batch * head, length, bucket_length * 4, rounds]

        flattened_value = value.flatten(0, 1)[..., None].expand(-1, -1, -1, self.rounds)
        # [batch * head, length, d_k, rounds]
        reordered_value = torch.gather(flattened_value, dim=1, index=expanded_hash_indices)
        # [batch * head, length, d_k, rounds]
        reshaped_value = reordered_value.reshape(-1, self.n_buckets // 2, bucket_length * 2, self.d_k, self.rounds)
        # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
        lookback_value = look_back(reshaped_value)
        # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]

        attn_indices = hash_indices[..., None, :].expand(-1, -1, bucket_length * 4, -1)
        # [batch * head, length, bucket_length * 4, rounds]
        reordered_p_attn = torch.gather(p_attn, dim=1, index=attn_indices)
        # [batch * head, length, bucket_length * 4, rounds]
        new_p_attn = reordered_p_attn.reshape(-1, self.n_buckets // 2, bucket_length * 2, bucket_length * 4, self.rounds)
        # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]

        attention = torch.einsum('...ijl,...jkl->...ikl', new_p_attn, lookback_value)
        # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
        attention = attention.flatten(1, 2)
        # [batch * head, length, d_k, rounds]
        new_indices = original_indices[..., None, :].expand(-1, -1, self.d_k, -1)
        # [batch * head, length, d_k, rounds]
        attention = torch.gather(attention, dim=1, index=new_indices).sum(dim=-1)
        # [batch * head, length, d_k]
        attention = attention.reshape(-1, self.head, length, self.d_k)
        # [batch, head, length, d_k]

        return attention

class MultiRoundLSHAttention(nn.Module):
    def __init__(self, hp, args):
        super(MultiRoundLSHAttention, self).__init__()
        self.d_k = hp.model.d_model // hp.model.head
        self.head = hp.model.head
        self.n_buckets = hp.model.n_buckets
        self.rounds = hp.model.rounds
        self.linear_query = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_value = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_out = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.lshattention = LSHAttention(hp, args)

    def forward(self, query, value, mask, seed, random=True):
        length = query.size(1)

        query = self.linear_query(query).reshape(-1, length, self.head, self.d_k).transpose(1, 2)
        # [batch, head, length, d_k]
        value = self.linear_value(value).reshape(-1, length, self.head, self.d_k).transpose(1, 2)
        # [batch, head, length, d_k]

        x = self.lshattention(query, value, mask, seed, random)
        # [batch, head, length, d_k]

        x = x.transpose(1, 2).flatten(-2, -1)
        # [batch, length, d_model]

        return self.linear_out(x)
