import os
import json
import random
import csv
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from urllib.parse import quote
from urllib.request import Request, urlopen


def authenticate_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi

    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        raise EnvironmentError(
            "Missing KAGGLE_USERNAME or KAGGLE_KEY environment variables. "
            "Set them in Colab before running dataset download."
        )

    api = KaggleApi()
    api.authenticate()
    return api


def download_and_extract_kaggle_dataset(dataset_slug, download_dir, extract_dir, force=False):
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    safe_slug = dataset_slug.replace("/", "__")
    marker = os.path.join(extract_dir, f".extracted_{safe_slug}.done")
    if os.path.exists(marker) and not force:
        print(f"[DATA] Reusing extracted dataset at: {extract_dir}")
        return extract_dir

    print(f"[DATA] Downloading dataset: {dataset_slug}")
    api = authenticate_kaggle()
    api.dataset_download_files(dataset=dataset_slug, path=download_dir, unzip=False, quiet=False)

    zip_name = dataset_slug.split("/")[-1] + ".zip"
    zip_path = os.path.join(download_dir, zip_name)
    if not os.path.exists(zip_path):
        zips = [f for f in os.listdir(download_dir) if f.endswith(".zip")]
        if not zips:
            raise FileNotFoundError("No dataset zip file found after Kaggle download.")
        zips.sort(key=lambda x: os.path.getsize(os.path.join(download_dir, x)), reverse=True)
        zip_path = os.path.join(download_dir, zips[0])

    print(f"[DATA] Extracting zip: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    with open(marker, "w", encoding="utf-8") as f:
        f.write("done")

    return extract_dir


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def _hf_fetch_rows(dataset_name, config_name, split_name, offset, length, timeout=60):
    base = "https://datasets-server.huggingface.co/rows"
    query = (
        f"dataset={quote(dataset_name, safe='')}&"
        f"config={quote(config_name, safe='')}&"
        f"split={quote(split_name, safe='')}&"
        f"offset={int(offset)}&length={int(length)}"
    )
    url = f"{base}?{query}"
    req = Request(url, headers={"User-Agent": "slm-trainer/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(payload)
    return data.get("rows", [])


def _extract_messages_text(row_obj):
    if not isinstance(row_obj, dict):
        return ""

    messages = row_obj.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            messages = [messages]

    if not isinstance(messages, list):
        return ""

    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, list):
                joined = []
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("text", "")
                        if t:
                            joined.append(str(t))
                    elif c is not None:
                        joined.append(str(c))
                content = " ".join(joined)
            text = clean_text(content)
            if text:
                parts.append(text)
        elif msg is not None:
            text = clean_text(str(msg))
            if text:
                parts.append(text)

    return " ".join(parts)


def build_ultrachat_messages_corpus(
    dataset_name,
    corpus_path,
    max_chars=60_000_000,
    config_name="default",
    split_candidates=None,
    page_size=100,
):
    if split_candidates is None:
        split_candidates = ["train_sft", "train", "train_gen", "test_sft", "test"]

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    chosen_split = None
    for sp in split_candidates:
        try:
            rows = _hf_fetch_rows(dataset_name, config_name, sp, offset=0, length=1)
            if rows is not None:
                chosen_split = sp
                break
        except Exception:
            continue

    if chosen_split is None:
        raise RuntimeError(
            f"Could not access HF dataset rows for {dataset_name}. "
            f"Tried splits: {split_candidates}"
        )

    print(f"[DATA] Using HF dataset={dataset_name}, config={config_name}, split={chosen_split}")
    print(f"[DATA] Building corpus from messages column (target chars={max_chars:,})")

    lines = []
    chars = 0
    offset = 0

    pbar = tqdm(total=max_chars, desc="HF messages -> corpus chars", leave=False)

    while chars < max_chars:
        rows = _hf_fetch_rows(dataset_name, config_name, chosen_split, offset=offset, length=page_size)
        if not rows:
            break

        consumed = 0
        for item in rows:
            row_obj = item.get("row", item)
            text = _extract_messages_text(row_obj)
            if not text:
                consumed += 1
                continue

            lines.append(text)
            n = len(text) + 1
            chars += n
            pbar.update(min(n, max_chars - pbar.n))
            consumed += 1

            if chars >= max_chars:
                break

        offset += consumed
        pbar.set_postfix(rows=f"{offset:,}", lines=f"{len(lines):,}")

        if consumed == 0:
            break

    pbar.close()

    if not lines:
        raise ValueError("No usable messages found in Hugging Face dataset.")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    return corpus_path, len(lines), chars


def _find_candidate_files(root_dir):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for fn in filenames:
            lower = fn.lower()
            if lower.endswith(".csv") or lower.endswith(".txt") or lower.endswith(".jsonl"):
                files.append(os.path.join(root, fn))
    return files


def _extract_text_from_csv(file_path, max_chars, language_column="language", lang_value="en"):
    collected = []
    total_chars = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.lower() for c in (reader.fieldnames or [])]

        text_candidates = [
            "text",
            "lyrics",
            "content",
            "sentence",
            "review",
            "comment",
            "body",
            "title",
        ]
        text_cols = [c for c in (reader.fieldnames or []) if c.lower() in text_candidates]
        if not text_cols and reader.fieldnames:
            text_cols = [reader.fieldnames[0]]

        lang_col_exact = None
        if reader.fieldnames:
            for c in reader.fieldnames:
                if c.lower() == language_column.lower():
                    lang_col_exact = c
                    break

        row_iter = tqdm(reader, desc=f"CSV {os.path.basename(file_path)}", leave=False)
        for row in row_iter:
            if lang_col_exact is not None:
                lang = str(row.get(lang_col_exact, "")).lower().strip()
                if lang and lang != lang_value:
                    continue

            text_parts = []
            for c in text_cols:
                v = clean_text(row.get(c, ""))
                if v:
                    text_parts.append(v)

            if text_parts:
                line = " ".join(text_parts)
                collected.append(line)
                total_chars += len(line) + 1
                row_iter.set_postfix(chars=f"{total_chars:,}")
                if total_chars >= max_chars:
                    break

    return collected, total_chars


def build_english_corpus(extract_dir, corpus_path, max_chars=120_000_000):
    files = _find_candidate_files(extract_dir)
    if not files:
        raise FileNotFoundError("No candidate text/csv/jsonl files found in extracted dataset.")

    print(f"[DATA] Building corpus from {len(files)} files (target chars={max_chars:,})")
    lines = []
    chars = 0

    file_iter = tqdm(files, desc="Corpus files", leave=False)
    for fp in file_iter:
        lower = fp.lower()
        file_iter.set_postfix(processed_chars=f"{chars:,}")
        if lower.endswith(".csv"):
            extracted, n = _extract_text_from_csv(fp, max_chars=max_chars - chars)
            lines.extend(extracted)
            chars += n
        elif lower.endswith(".txt"):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in tqdm(f, desc=f"TXT {os.path.basename(fp)}", leave=False):
                    line = clean_text(line)
                    if not line:
                        continue
                    lines.append(line)
                    chars += len(line) + 1
                    if chars >= max_chars:
                        break
        elif lower.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in tqdm(f, desc=f"JSONL {os.path.basename(fp)}", leave=False):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    text = clean_text(obj.get("text", ""))
                    if not text:
                        continue
                    lang = str(obj.get("language", "en")).lower().strip()
                    if lang and lang != "en":
                        continue
                    lines.append(text)
                    chars += len(text) + 1
                    if chars >= max_chars:
                        break

        if chars >= max_chars:
            break

    if not lines:
        raise ValueError("No usable English text found in dataset.")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    return corpus_path, len(lines), chars


def prepare_train_val_tokens(corpus_path, tokenizer, train_tokens_path, val_tokens_path, train_split=0.9, force_retokenize=False):
    if (not force_retokenize) and os.path.exists(train_tokens_path) and os.path.exists(val_tokens_path):
        print("[TOK] Reusing existing train/val token files")
        train_ids = np.load(train_tokens_path, mmap_mode="r")
        val_ids = np.load(val_tokens_path, mmap_mode="r")
        return int(train_ids.shape[0]), int(val_ids.shape[0])

    print("[TOK] Tokenizing corpus into train/val arrays...")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        line_iter = tqdm(f, desc="Tokenizing corpus lines", leave=False)
        cleaned_lines = (clean_text(line) for line in line_iter)
        ids = tokenizer.encode_lines_to_numpy(
            cleaned_lines,
            add_bos=True,
            add_eos=True,
            dtype=np.int32,
            merge_every=4096,
            show_progress=False,
        )
    split_idx = int(len(ids) * train_split)
    split_idx = max(2, min(split_idx, len(ids) - 2))

    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    np.save(train_tokens_path, train_ids)
    np.save(val_tokens_path, val_ids)

    return train_ids.shape[0], val_ids.shape[0]


class LanguageModelingDataset(Dataset):
    def __init__(self, tokens_path, context_length):
        tokens = np.load(tokens_path, mmap_mode="r")
        self.context_length = context_length

        usable_tokens = (len(tokens) // context_length) * context_length
        tokens = tokens[:usable_tokens]

        self.tokens = tokens.reshape(-1, context_length)
        self.length = self.tokens.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.tokens[idx]
        y = np.roll(x, -1)
        y[-1] = 0
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def create_dataloaders(train_tokens_path, val_tokens_path, context_length, batch_size, eval_batch_size):
    train_ds = LanguageModelingDataset(train_tokens_path, context_length)
    val_ds = LanguageModelingDataset(val_tokens_path, context_length)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
