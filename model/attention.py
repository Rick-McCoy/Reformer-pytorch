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

def look_back(x):
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
    size = x.size()
    new_size = size[:1] + (1, ) + size[2:]
    pad = torch.cat([torch.zeros(new_size, dtype=x.dtype).cuda(non_blocking=True), x], dim=1)
    # [batch * head, n_buckets // 2 + 1, bucket_length * 2, d_k, rounds]
    concat = torch.cat([pad[:, :-1], pad[:, 1:]], dim=2)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]
    return concat

def lshattention(query, key, value, rounds, n_buckets, mask, dropout=None):
    head = query.size(1)
    d_k = query.size(-1)
    length = query.size(-2)
    bucket_length = length // n_buckets

    flattened_query = query.flatten(0, 1)[..., None].expand(-1, -1, -1, rounds)
    # [batch * head, length, d_k, rounds]
    flattened_key = key.flatten(0, 1)[..., None].expand(-1, -1, -1, rounds)
    # [batch * head, length, d_k, rounds]
    flattened_value = value.flatten(0, 1)[..., None].expand(-1, -1, -1, rounds)
    # [batch * head, length, d_k, rounds]

    hashes = localitysensitivehash(flattened_query[..., 0], d_k, n_buckets, rounds)
    # [batch * head, length, rounds]
    hash_indices = torch.argsort(hashes, dim=1)
    # [batch * head, length, rounds]
    expanded_hash_indices = hash_indices[:, :, None, :].expand(-1, -1, d_k, -1)
    # [batch * head, length, d_k, rounds]

    reordered_query = torch.gather(flattened_query, dim=1, index=expanded_hash_indices)
    # [batch * head, length, d_k, rounds]
    reordered_query = reordered_query.reshape(-1, n_buckets // 2, bucket_length * 2, d_k, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]

    reordered_key = torch.gather(flattened_key, dim=1, index=expanded_hash_indices)
    # [batch * head, length, d_k, rounds]
    reordered_key = reordered_key.reshape(-1, n_buckets // 2, bucket_length * 2, d_k, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]

    lookback_key = look_back(reordered_key)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]

    scores = torch.einsum('...ijk,...ljk->...ilk', reordered_query, lookback_key) / math.sqrt(d_k)
    # [batch * head, n_buckets, bucket_length * 2, bucket_length * 4, rounds]
    scores = scores.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]

    original_indices = torch.argsort(hash_indices, dim=1)
    # [batch * head, length, rounds]
    score_indices = original_indices[..., None, :].expand(-1, -1, bucket_length * 4, -1)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = torch.gather(scores, dim=1, index=score_indices)
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
    original_mask = torch.gather(flattened_mask, dim=1, index=score_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - ~original_mask.detach() * 1e9

    hashes = torch.gather(hashes, dim=1, index=hash_indices)
    # [batch * head, length, rounds]
    hashes = hashes.reshape(-1, n_buckets // 2, bucket_length * 2, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
    key_hash = look_back(hashes)
    # [batch * head, n_buckets // 2, bucket_length * 4, rounds]
    hash_equiv_mask = (hashes[..., None, :] == key_hash[..., None, :, :])
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    hash_equiv_mask = hash_equiv_mask.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    hash_equiv_mask = torch.gather(hash_equiv_mask, dim=1, index=score_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - ~hash_equiv_mask.detach() * 1e9

    reshaped_mask_indices = original_indices.reshape(-1, n_buckets // 2, bucket_length * 2, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
    lookback_mask_indices = look_back(reshaped_mask_indices)
    # [batch * head, n_buckets // 2, bucket_length * 4, rounds]
    query_indices = reshaped_mask_indices
    # [batch * head, n_buckets // 2, bucket_length * 2, rounds]
    key_indices = lookback_mask_indices
    # [batch * head, n_buckets // 2, bucket_length * 4, rounds]

    causal_mask = query_indices[..., None, :] > key_indices[..., None, :, :]
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    causal_mask = causal_mask.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    causal_mask = torch.gather(causal_mask, dim=1, index=score_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - ~causal_mask.detach() * 1e9

    indice_equiv_mask = query_indices[..., None, :] == key_indices[..., None, :, :]
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    indice_equiv_mask = indice_equiv_mask.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    indice_equiv_mask = torch.gather(indice_equiv_mask, dim=1, index=score_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - indice_equiv_mask.detach() * 1e5

    expanded_key_indices = key_indices[..., None, :, :].expand(-1, -1, bucket_length * 2, -1, -1)
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, rounds]
    flattened_key_indices = expanded_key_indices.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, rounds]
    reordered_key_indices = torch.gather(flattened_key_indices, dim=1, index=score_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    count_repeat_key = torch.zeros_like(reordered_key_indices)
    # [batch * head, length, bucket_length * 4, rounds]
    for split_key_indices in torch.chunk(reordered_key_indices, chunks=bucket_length * 4, dim=-2):
        count_repeat_key += (reordered_key_indices[..., None] == split_key_indices[..., None, :]).sum(dim=-1)
    # [batch * head, length, bucket_length * 4, rounds]
    scores = scores - count_repeat_key.float().log().detach()

    scores = scores.flatten(-2, -1)
    # [batch * head, length, bucket_length * 4 * rounds]
    p_attn = F.softmax(scores, dim=-1)
    # [batch * head, length, bucket_length * 4 * rounds]

    if dropout is not None:
        p_attn = dropout(p_attn)

    reordered_value = torch.gather(flattened_value, dim=1, index=expanded_hash_indices)
    # [batch * head, length, d_k, rounds]
    reshaped_value = reordered_value.reshape(-1, n_buckets // 2, bucket_length * 2, d_k, rounds)
    # [batch * head, n_buckets // 2, bucket_length * 2, d_k, rounds]
    lookback_value = look_back(reshaped_value)
    # [batch * head, n_buckets // 2, bucket_length * 4, d_k, rounds]
    repeat_value = lookback_value[:, :, None, ...].expand(-1, -1, bucket_length * 2, -1, -1, -1)
    # [batch * head, n_buckets // 2, bucket_length * 2, bucket_length * 4, d_k, rounds]
    repeat_value = repeat_value.flatten(1, 2)
    # [batch * head, length, bucket_length * 4, d_k, rounds]
    value_indices = score_indices[..., None, :].expand(-1, -1, -1, d_k, -1)
    # [batch * head, length, bucket_length * 4, d_k, rounds]
    original_value = torch.gather(repeat_value, dim=1, index=value_indices)
    # [batch * head, length, bucket_length * 4, d_k, rounds]
    original_value = original_value.transpose(-2, -1).flatten(-3, -2)
    # [batch * head, length, bucket_length * 4 * rounds, d_k]

    attention = torch.einsum('...i,...ij->...j', p_attn, original_value)
    # [batch * head, length, d_k]
    attention = attention.reshape(-1, head, length, d_k)
    # [batch, head, length, d_k]

    return attention, p_attn.reshape(-1, head, length, bucket_length * 4 * rounds)

def localitysensitivehash(inp, d_k, n_buckets, rounds):
    rand_matrix = torch.rand([d_k, rounds, n_buckets // 2]).cuda(non_blocking=True)
    rand_matrix = rand_matrix / torch.norm(rand_matrix, dim=-1, keepdim=True)
    x = torch.einsum('...i,ijk->...jk', inp, rand_matrix)
    return torch.argmax(torch.cat([x, -x], dim=-1), dim=-1)

class MultiRoundLSHAttention(nn.Module):
    def __init__(self, hp, args):
        super(MultiRoundLSHAttention, self).__init__()
        self.d_k = hp.model.d_model // hp.model.head
        self.head = hp.model.head
        self.n_buckets = hp.model.n_buckets
        self.rounds = hp.model.rounds
        self.linear_query = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_key = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_value = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.linear_out = nn.Linear(hp.model.d_model, hp.model.d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=hp.model.dropout)
    
    def forward(self, query, key, value, mask):
        length = query.size(1)

        query = self.linear_query(query).reshape(-1, length, self.head, self.d_k).transpose(1 ,2)
        # [batch, head, length, d_k]
        key = query / torch.norm(query, dim=-1, keepdim=True)
        # [batch, head, length, d_k]
        value = self.linear_value(value).reshape(-1, length, self.head, self.d_k).transpose(1 ,2)
        # [batch, head, length, d_k]

        x, self.attn = lshattention(
            query,
            key,
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
