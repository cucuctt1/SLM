import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _shape(self, x, bsz, seq_len):
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        bsz, tgt_len, _ = x.shape

        q = self._shape(self.q_proj(x), bsz, tgt_len)
        k = self._shape(self.k_proj(x), bsz, tgt_len)
        v = self._shape(self.v_proj(x), bsz, tgt_len)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        src_len = k.size(2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        q_positions = torch.arange(tgt_len, device=x.device).unsqueeze(-1)
        k_positions = torch.arange(src_len, device=x.device).unsqueeze(0)
        if past_kv is None:
            causal = k_positions <= q_positions
        else:
            past_len = past_kv[0].size(2)
            causal = k_positions <= (q_positions + past_len)
        causal = causal.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            if mask.size(-1) != src_len:
                if mask.size(-1) < src_len:
                    pad = torch.ones((bsz, 1, 1, src_len - mask.size(-1)), device=x.device, dtype=torch.bool)
                    mask = torch.cat([pad, mask], dim=-1)
                else:
                    mask = mask[..., -src_len:]
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, tgt_len, self.d_model)
        out = self.resid_dropout(self.out_proj(context))

        present_kv = (k, v) if use_cache else None
        return out, present_kv
