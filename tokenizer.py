import os
import json
import numpy as np
from tqdm import tqdm

from bpe import train_byte_level_bpe, apply_bpe_to_word, save_bpe_files, load_bpe_files


class ByteLevelBPETokenizer:
    def __init__(self, vocab_size=50000, context_length=512):
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_id = None
        self.bos_id = None
        self.eos_id = None
        self.unk_id = None

        self.token_to_bytes = {}
        self.byte_string_to_id = {}
        self.merges = []
        self.merges_ranked = {}
        self.vocab = {}
        self.id_to_token = {}
        self.eow_id = 256
        self.word_cache = {}
        self.max_word_cache_size = 200_000

    def _build_vocab_maps(self):
        self.vocab = {}
        self.id_to_token = {}

        for k, byte_seq in self.token_to_bytes.items():
            tid = int(k)
            tok = "bytes:" + " ".join([f"{b:02x}" for b in byte_seq])
            self.vocab[tok] = tid
            self.id_to_token[str(tid)] = tok

        self.pad_id = self.vocab_size - 4
        self.bos_id = self.vocab_size - 3
        self.eos_id = self.vocab_size - 2
        self.unk_id = self.vocab_size - 1

        for tid in range(257, self.pad_id):
            key = str(tid)
            if key not in self.token_to_bytes:
                self.token_to_bytes[key] = []
                tok = f"extra:{tid}"
                self.vocab[tok] = tid
                self.id_to_token[key] = tok

        special_map = {
            self.pad_token: self.pad_id,
            self.bos_token: self.bos_id,
            self.eos_token: self.eos_id,
            self.unk_token: self.unk_id,
        }
        for tok, tid in special_map.items():
            self.vocab[tok] = tid
            self.id_to_token[str(tid)] = tok

        if len(self.vocab) > self.vocab_size:
            raise ValueError(f"Tokenizer vocab ({len(self.vocab)}) exceeded configured vocab_size ({self.vocab_size}).")

        if len(self.vocab) < self.vocab_size:
            current_ids = set(self.vocab.values())
            for tid in range(self.vocab_size):
                if tid in current_ids:
                    continue
                tok = f"extra:{tid}"
                self.vocab[tok] = tid
                self.id_to_token[str(tid)] = tok
                key = str(tid)
                if key not in self.token_to_bytes:
                    self.token_to_bytes[key] = []
                if len(self.vocab) == self.vocab_size:
                    break

    def train(self, text):
        bpe_state = train_byte_level_bpe(
            text=text,
            target_vocab_size=self.vocab_size,
            reserved_special=4,
            min_pair_freq=2,
            show_progress=True,
        )
        self.merges = bpe_state["merges"]
        self.token_to_bytes = bpe_state["token_to_bytes"]
        self.eow_id = bpe_state["eow_id"]

        self.merges_ranked = {}
        for rank, m in enumerate(self.merges):
            pair = tuple(m["pair"])
            self.merges_ranked[pair] = (rank, m["new_id"])

        self.byte_string_to_id = {}
        for k, byte_seq in self.token_to_bytes.items():
            self.byte_string_to_id[bytes(byte_seq)] = int(k)

        self._build_vocab_maps()

    def encode(self, text, add_special_tokens=True, max_length=None, padding=False, truncation=True):
        if max_length is None:
            max_length = self.context_length

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_id)

        words = [w for w in text.split() if w]
        for w in words:
            bpe_ids = self.word_cache.get(w)
            if bpe_ids is None:
                bpe_ids = apply_bpe_to_word(w, self.merges_ranked, eow_id=self.eow_id)
                if len(self.word_cache) >= self.max_word_cache_size:
                    self.word_cache.clear()
                self.word_cache[w] = bpe_ids
            for tid in bpe_ids:
                if tid < self.vocab_size:
                    token_ids.append(tid)
                else:
                    token_ids.append(self.unk_id)

        if add_special_tokens:
            token_ids.append(self.eos_id)

        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [self.pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }

    def decode(self, token_ids, skip_special_tokens=True):
        words = []
        current_bytes = []

        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue

            if tid == self.unk_id:
                if current_bytes:
                    words.append(bytes(current_bytes).decode("utf-8", errors="ignore"))
                    current_bytes = []
                words.append("<unk>")
                continue

            byte_seq = self.token_to_bytes.get(str(int(tid)), [])
            if len(byte_seq) == 0:
                if current_bytes:
                    words.append(bytes(current_bytes).decode("utf-8", errors="ignore"))
                    current_bytes = []
            else:
                current_bytes.extend(byte_seq)

        if current_bytes:
            words.append(bytes(current_bytes).decode("utf-8", errors="ignore"))

        return " ".join([w for w in words if w])

    def encode_batch(self, texts, max_length=None, padding=True, truncation=True):
        input_ids = []
        attention_masks = []
        for t in texts:
            out = self.encode(
                t,
                add_special_tokens=True,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )
            input_ids.append(out["input_ids"])
            attention_masks.append(out["attention_mask"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }

    def encode_lines_to_numpy(self, lines, add_bos=True, add_eos=True, dtype=np.int32, merge_every=4096, show_progress=False, total_lines=None):
        chunks = []
        if add_bos:
            chunks.append(np.array([self.bos_id], dtype=dtype))

        pending = []
        iterator = tqdm(lines, total=total_lines, desc="Tokenizing corpus", leave=False) if show_progress else lines
        for line in iterator:
            if not line:
                continue
            encoded = self.encode(
                line,
                add_special_tokens=False,
                max_length=10_000_000_000,
                padding=False,
                truncation=False,
            )
            ids = encoded["input_ids"]
            if not ids:
                continue
            pending.append(np.asarray(ids, dtype=dtype))

            if len(pending) >= merge_every:
                chunks.append(np.concatenate(pending, axis=0))
                pending = []

        if pending:
            chunks.append(np.concatenate(pending, axis=0))

        if add_eos:
            chunks.append(np.array([self.eos_id], dtype=dtype))

        if not chunks:
            return np.array([], dtype=dtype)

        if len(chunks) == 1:
            return chunks[0]

        return np.concatenate(chunks, axis=0)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        merges_dump = self.merges
        vocab_dump = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "eow_id": self.eow_id,
            "token_to_bytes": self.token_to_bytes,
            "vocab": self.vocab,
            "id_to_token": self.id_to_token,
        }
        save_bpe_files(output_dir, vocab_dump, merges_dump)

    def load(self, output_dir):
        vocab_dump, merges_dump = load_bpe_files(output_dir)
        self.vocab_size = vocab_dump["vocab_size"]
        self.context_length = vocab_dump["context_length"]
        self.pad_id = vocab_dump["pad_id"]
        self.bos_id = vocab_dump["bos_id"]
        self.eos_id = vocab_dump["eos_id"]
        self.unk_id = vocab_dump["unk_id"]
        self.eow_id = vocab_dump["eow_id"]
        self.token_to_bytes = vocab_dump["token_to_bytes"]
        self.vocab = vocab_dump["vocab"]
        self.id_to_token = vocab_dump["id_to_token"]
        self.merges = merges_dump

        self.merges_ranked = {}
        for rank, m in enumerate(self.merges):
            pair = tuple(m["pair"])
            self.merges_ranked[pair] = (rank, m["new_id"])

        self.byte_string_to_id = {}
        for k, byte_seq in self.token_to_bytes.items():
            self.byte_string_to_id[bytes(byte_seq)] = int(k)
        self.word_cache = {}

    def try_load(self, output_dir):
        vocab_path = os.path.join(output_dir, "vocab.json")
        merges_path = os.path.join(output_dir, "merges.json")
        if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
            return False
        self.load(output_dir)
        if len(self.vocab) != self.vocab_size:
            return False
        return True

    def get_state(self):
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "eow_id": self.eow_id,
            "token_to_bytes": self.token_to_bytes,
            "vocab": self.vocab,
            "id_to_token": self.id_to_token,
            "merges": self.merges,
        }

    def set_state(self, state):
        self.vocab_size = state["vocab_size"]
        self.context_length = state["context_length"]
        self.pad_id = state["pad_id"]
        self.bos_id = state["bos_id"]
        self.eos_id = state["eos_id"]
        self.unk_id = state["unk_id"]
        self.eow_id = state["eow_id"]
        self.token_to_bytes = state["token_to_bytes"]
        self.vocab = state["vocab"]
        self.id_to_token = state["id_to_token"]
        self.merges = state["merges"]

        self.merges_ranked = {}
        for rank, m in enumerate(self.merges):
            pair = tuple(m["pair"])
            self.merges_ranked[pair] = (rank, m["new_id"])

        self.byte_string_to_id = {}
        for k, byte_seq in self.token_to_bytes.items():
            self.byte_string_to_id[bytes(byte_seq)] = int(k)
