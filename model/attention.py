import math
import torch
import torch.nn.functional as F

from torch import nn

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // head
        self.head = head
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1)
        batch = query.size(0)

        query = self.linear_query(query).reshape(batch, -1, self.head, self.d_k).transpose(1 ,2)
        key = self.linear_key(key).reshape(batch, -1, self.head, self.d_k).transpose(1 ,2)
        value = self.linear_value(value).reshape(batch, -1, self.head, self.d_k).transpose(1 ,2)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).reshape(batch, -1, self.head * self.d_k)

        return self.linear_out(x)

def get_count(x, y):
    count = torch.zeros_like(x)
    for split in torch.split(x, split_size_or_sections=1, dim=-1):
        count += (y == split)
    return count.squeeze(-1)

def look_back(x, zeros=True) -> torch.Tensor:
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
    size = x.size()
    new_size = size[:1] + (1, ) + size[2:]
    if zeros:
        pad = torch.cat([x.new_zeros(new_size, dtype=x.dtype), x], dim=1)
    else:
        pad = torch.cat([x.new_empty(new_size).fill_(1e9).type_as(x), x], dim=1)
    # [batch * head, n_buckets // 2 + 1, bucket_length * 2, d_k, rounds]
    concat = torch.cat([pad[:, :-1], pad[:, 1:]], dim=2)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]
    return concat

def lshattention(query, value, rounds, n_buckets, mask, dropout):
    _, head, length, d_k = query.size()
    bucket_length = length // n_buckets

    query = query / torch.norm(query, dim=-1, keepdim=True)

    flattened_query = query.flatten(0, 1)
    # [batch * head, length, d_k]

    hashes = localitysensitivehash(flattened_query, d_k, n_buckets, rounds)
    # [batch * head, length, rounds]
    sorted_hashes, hash_indices = torch.sort(hashes, dim=1)
    # [batch * head, length, rounds]
    expanded_hash_indices = hash_indices[:, :, None, :].expand(-1, -1, d_k, -1)
    # [batch * head, length, d_k, rounds]

    expanded_query = flattened_query[..., None].expand(-1, -1, -1, rounds)
    # [batch * head, length, d_k, rounds]
    reordered_query = torch.gather(expanded_query, dim=1, index=expanded_hash_indices)
    # [batch * head, length, d_k, rounds]
    reordered_query = reordered_query.reshape(-1, n_buckets // 2, bucket_length * 2, d_k, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]

    lookback_key = look_back(reordered_query)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]

    scores = torch.einsum('...ijk,...ljk->...ilk', reordered_query, lookback_key) / math.sqrt(d_k)
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    scores = scores.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]

    mask = mask[:, None, :, None].expand(-1, head, -1, rounds).flatten(0, 1)
    # [batch * head, length, rounds]
    reordered_mask = torch.gather(mask, dim=1, index=hash_indices)
    # [batch * head, length, rounds]
    reordered_mask = reordered_mask.reshape(-1, n_buckets // 2, bucket_length * 2, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
    lookback_mask = look_back(reordered_mask)
    # [batch * head, n_buckets // 2, bucket_length * 4, rounds]
    reshaped_mask = lookback_mask[..., None, :, :].expand(-1, -1, bucket_length * 2, -1, -1)
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    flattened_mask = reshaped_mask.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - ~flattened_mask.detach() * 1e9

    sorted_hashes = sorted_hashes.reshape(-1, n_buckets // 2, bucket_length * 2, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
    lookback_hash = look_back(sorted_hashes, False)
    # [batch * head, n_buckets // 2, bucket_length * 4, rounds]
    hash_equiv_mask = (sorted_hashes[..., None, :] == lookback_hash[..., None, :, :])
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    hash_equiv_mask = hash_equiv_mask.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - ~hash_equiv_mask.detach() * 1e9

    query_indices = hash_indices.reshape(-1, n_buckets // 2, bucket_length * 2, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
    key_indices = look_back(query_indices, False)
    # [batch * head, n_buckets // 2, bucket_length * 4, rounds]

    causal_mask = query_indices[..., None, :] < key_indices[..., None, :, :]
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    causal_mask = causal_mask.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - causal_mask.detach() * 1e9

    indice_equiv_mask = query_indices[..., None, :] == key_indices[..., None, :, :]
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    indice_equiv_mask = indice_equiv_mask.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - indice_equiv_mask.detach() * 1e5

    original_indices = torch.argsort(hash_indices, dim=1)
    # [batch * head, length, rounds]
    score_indices = original_indices[..., None, :].expand(-1, -1, bucket_length * 4, -1)
    # [batch * head, length, bucket_length * 4, rounds]

    expanded_key_indices = key_indices[..., None, :, :].expand(-1, -1, bucket_length * 2, -1, -1)
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    flattened_key_indices = expanded_key_indices.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    reordered_key_indices = torch.gather(flattened_key_indices, dim=1, index=score_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    count_repeat_key = torch.ones_like(reordered_key_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    for i, i1 in enumerate(torch.split(reordered_key_indices[..., :-1], split_size_or_sections=1, dim=-1)):
        for j, i2 in enumerate(torch.split(reordered_key_indices[..., i + 1:], split_size_or_sections=1, dim=-1)):
            comp = get_count(i1, i2)
            count_repeat_key[..., i] += comp
            count_repeat_key[..., j] += comp
    # for split_key_indices in torch.split(reordered_key_indices, split_size_or_sections=1, dim=-2):
    #     count_repeat_key += (reordered_key_indices[..., None] == split_key_indices[..., None, :]).sum(dim=-1)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = torch.gather(scores, dim=1, index=score_indices)
    scores = scores - count_repeat_key.float().log().detach()

    scores = scores.flatten(-2, -1)
    # [batch * head, length, bucket_length * 4 * rounds]
    p_attn = F.softmax(scores, dim=-1)
    # [batch * head, length, bucket_length * 4 * rounds]

    p_attn = dropout(p_attn)

    flattened_value = value.flatten(0, 1)[..., None].expand(-1, -1, -1, rounds)
    # [batch * head, length, d_k, rounds]
    reordered_value = torch.gather(flattened_value, dim=1, index=expanded_hash_indices)
    # [batch * head, length, d_k, rounds]
    reshaped_value = reordered_value.reshape(-1, n_buckets // 2, bucket_length * 2, d_k, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
    lookback_value = look_back(reshaped_value)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]

    attn_indices = hash_indices[..., None, :].expand(-1, -1, bucket_length * 4, -1).flatten(-2, -1)
    # [batch * head, length, bucket_length * 4 * rounds]
    reordered_p_attn = torch.gather(p_attn, dim=1, index=attn_indices)
    new_p_attn = reordered_p_attn.reshape(-1, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]

    attention = torch.einsum('...ijl,...jkl->...ikl', new_p_attn, lookback_value)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
    attention = attention.flatten(1, 2)
    # [batch * head, length, d_k, rounds]
    new_indices = original_indices[..., None, :].expand(-1, -1, d_k, -1)
    # [batch * head, length, d_k, rounds]
    attention = torch.gather(attention, dim=1, index=new_indices).sum(dim=-1)
    # [batch * head, length, d_k]
    attention = attention.reshape(-1, head, length, d_k)
    # [batch, head, length, d_k]

    return attention

def localitysensitivehash(inp, d_k, n_buckets, rounds):
    # [batch * head, length, d_k]
    rand_matrix = torch.rand([d_k, rounds, n_buckets // 2]).cuda(non_blocking=True)
    # [d_k, rounds, n_buckets // 2]
    rand_matrix = rand_matrix / torch.norm(rand_matrix, dim=-1, keepdim=True)
    # [d_k, rounds, n_buckets // 2]
    x = torch.einsum('...i,ijk->...jk', inp, rand_matrix)
    # [batch * head, length, rounds, n_buckets // 2]
    return torch.argmax(torch.cat([x, -x], dim=-1), dim=-1)
    # [batch * head, length, rounds]

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
        self.dropout = nn.Dropout(p=hp.model.dropout)
    
    def forward(self, query, value, mask):
        length = query.size(1)

        query = self.linear_query(query).reshape(-1, length, self.head, self.d_k).transpose(1 ,2)
        # [batch, head, length, d_k]
        value = self.linear_value(value).reshape(-1, length, self.head, self.d_k).transpose(1 ,2)
        # [batch, head, length, d_k]

        x = lshattention(
            query,
            value,
            self.rounds,
            self.n_buckets,
            mask,
            dropout=self.dropout
        )
        # [batch, head, length, d_k], [batch, head, length, bucket_length * 4 * rounds]

        x = x.transpose(1, 2).reshape(-1, length, self.head * self.d_k)
        # [batch, length, d_model]

        return self.linear_out(x)
