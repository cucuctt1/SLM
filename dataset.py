import os
import json
import random
import csv
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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

    marker = os.path.join(extract_dir, ".extracted_done")
    if os.path.exists(marker) and not force:
        return extract_dir

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

        for row in reader:
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
                if total_chars >= max_chars:
                    break

    return collected, total_chars


def build_english_corpus(extract_dir, corpus_path, max_chars=120_000_000):
    files = _find_candidate_files(extract_dir)
    if not files:
        raise FileNotFoundError("No candidate text/csv/jsonl files found in extracted dataset.")

    lines = []
    chars = 0

    for fp in files:
        lower = fp.lower()
        if lower.endswith(".csv"):
            extracted, n = _extract_text_from_csv(fp, max_chars=max_chars - chars)
            lines.extend(extracted)
            chars += n
        elif lower.endswith(".txt"):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = clean_text(line)
                    if not line:
                        continue
                    lines.append(line)
                    chars += len(line) + 1
                    if chars >= max_chars:
                        break
        elif lower.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
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


def prepare_train_val_tokens(corpus_path, tokenizer, train_tokens_path, val_tokens_path, train_split=0.9):
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus_text = f.read()

    tokenized = tokenizer.encode(
        corpus_text,
        add_special_tokens=True,
        max_length=10_000_000_000,
        padding=False,
        truncation=False,
    )

    ids = np.array(tokenized["input_ids"], dtype=np.int32)
    split_idx = int(len(ids) * train_split)
    split_idx = max(2, min(split_idx, len(ids) - 2))

    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    np.save(train_tokens_path, train_ids)
    np.save(val_tokens_path, val_ids)

    return train_ids.shape[0], val_ids.shape[0]


class LanguageModelingDataset(Dataset):
    def __init__(self, tokens_path, context_length):
        self.tokens = np.load(tokens_path, mmap_mode="r")
        self.context_length = context_length
        self.length = max(1, len(self.tokens) - context_length - 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_length].astype(np.int64)
        y = self.tokens[idx + 1 : idx + 1 + self.context_length].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


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
