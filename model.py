import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from attention import MultiHeadSelfAttention
from feedforward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        h = self.ln_1(x)
        attn_out, present_kv = self.attn(h, attention_mask=attention_mask, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        h = self.ln_2(x)
        x = x + self.ffn(h)
        return x, present_kv


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    bias=cfg.bias,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, targets=None, past_kvs=None, use_cache=False):
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context length {self.cfg.context_length}")

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
            past_len = 0
        else:
            past_len = past_kvs[0][0].size(2) if past_kvs[0] is not None else 0

        pos = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(bsz, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        presents = []
        for i, block in enumerate(self.blocks):
            if self.cfg.grad_checkpointing and self.training and (not use_cache):
                def custom_forward(hidden_states):
                    out, _ = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_kv=None,
                        use_cache=False,
                    )
                    return out

                x = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
                present = None
            else:
                x, present = block(x, attention_mask=attention_mask, past_kv=past_kvs[i], use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss, (presents if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=128,
        temperature=1.0,
        top_k=0,
        eos_token_id=None,
        use_kv_cache=True,
    ):
        self.eval()
        generated = input_ids
        past_kvs = None

        for _ in range(max_new_tokens):
            if use_kv_cache and past_kvs is not None:
                idx_cond = generated[:, -1:]
            else:
                idx_cond = generated[:, -self.cfg.context_length :]

            logits, _, past_kvs = self(
                idx_cond,
                past_kvs=past_kvs if use_kv_cache else None,
                use_cache=use_kv_cache,
            )

            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

            if generated.size(1) >= self.cfg.context_length and not use_kv_cache:
                generated = generated[:, -self.cfg.context_length :]

        return generated
