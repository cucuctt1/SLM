import json
from collections import Counter


def _get_stats(corpus_words):
    stats = Counter()
    for w in corpus_words:
        if len(w) < 2:
            continue
        for i in range(len(w) - 1):
            stats[(w[i], w[i + 1])] += 1
    return stats


def _merge_pair(corpus_words, pair, new_id):
    a, b = pair
    new_corpus = []
    for w in corpus_words:
        if len(w) < 2:
            new_corpus.append(w)
            continue
        merged = []
        i = 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                merged.append(new_id)
                i += 2
            else:
                merged.append(w[i])
                i += 1
        new_corpus.append(merged)
    return new_corpus


def train_byte_level_bpe(text, target_vocab_size=50000, reserved_special=4, min_pair_freq=2):
    eow_id = 256
    base_vocab_size = 257
    max_bpe_tokens = max(base_vocab_size, target_vocab_size - reserved_special)

    words = [w for w in text.split() if w]
    corpus_words = []
    for w in words:
        ids = list(w.encode("utf-8", errors="ignore"))
        ids.append(eow_id)
        corpus_words.append(ids)

    merges = []
    token_to_bytes = {i: [i] for i in range(256)}
    token_to_bytes[eow_id] = []

    next_id = base_vocab_size

    while next_id < max_bpe_tokens:
        stats = _get_stats(corpus_words)
        if not stats:
            break
        best_pair, freq = stats.most_common(1)[0]
        if freq < min_pair_freq:
            break

        corpus_words = _merge_pair(corpus_words, best_pair, next_id)
        left_bytes = token_to_bytes.get(best_pair[0], [])
        right_bytes = token_to_bytes.get(best_pair[1], [])
        token_to_bytes[next_id] = left_bytes + right_bytes
        merges.append({"pair": [int(best_pair[0]), int(best_pair[1])], "new_id": int(next_id), "freq": int(freq)})
        next_id += 1

    return {
        "merges": merges,
        "token_to_bytes": {str(k): v for k, v in token_to_bytes.items()},
        "vocab_size_without_specials": next_id,
        "eow_id": eow_id,
    }


def apply_bpe_to_word(word, merges_ranked, eow_id=256):
    ids = list(word.encode("utf-8", errors="ignore")) + [eow_id]
    if not ids:
        return []

    changed = True
    while changed:
        changed = False
        best_idx = -1
        best_rank = None
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            rank = merges_ranked.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_idx = i
        if best_idx >= 0:
            pair = (ids[best_idx], ids[best_idx + 1])
            new_id = merges_ranked[pair][1]
            ids = ids[:best_idx] + [new_id] + ids[best_idx + 2 :]
            changed = True

    return [t for t in ids if t != eow_id]


def save_bpe_files(output_dir, vocab_json, merges):
    with open(f"{output_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    with open(f"{output_dir}/merges.json", "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False, indent=2)


def load_bpe_files(output_dir):
    with open(f"{output_dir}/vocab.json", "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    with open(f"{output_dir}/merges.json", "r", encoding="utf-8") as f:
        merges = json.load(f)
    return vocab_json, merges
