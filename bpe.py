import json
import os
import tempfile
import time
from collections import Counter
from tqdm import tqdm


def _get_stats(corpus_words):
    stats = Counter()
    for w, freq in corpus_words.items():
        if len(w) < 2:
            continue
        for i in range(len(w) - 1):
            stats[(w[i], w[i + 1])] += freq
    return stats


def _sentencepiece_like_pieces(text):
    words = [w for w in text.split() if w]
    return ["▁" + w for w in words]


def _merge_pair(corpus_words, pair, new_id):
    a, b = pair
    new_corpus = {}
    for w, freq in corpus_words.items():
        if len(w) < 2:
            new_corpus[w] = new_corpus.get(w, 0) + freq
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
        merged_t = tuple(merged)
        new_corpus[merged_t] = new_corpus.get(merged_t, 0) + freq
    return new_corpus


def _get_cpp_paths():
    base = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base, "bpe_fast.cpp")
    bin_path = os.path.join(base, "bpe_fast")
    return src, bin_path


def _compile_cpp_bpe(show_progress=True):
    src, bin_path = _get_cpp_paths()
    if not os.path.exists(src):
        return None

    if os.path.exists(bin_path):
        src_m = os.path.getmtime(src)
        bin_m = os.path.getmtime(bin_path)
        if bin_m >= src_m:
            if show_progress:
                print("C++ BPE binary is up to date, skipping compile.")
            return bin_path

    check = os.system("g++ --version > /dev/null 2>&1")
    if check != 0:
        return None

    compile_cmds = [
        ('g++ -O3 -std=c++17 -fopenmp "{src}" -o "{bin_path}"'.format(src=src, bin_path=bin_path), "openmp"),
        ('g++ -O3 -std=c++17 "{src}" -o "{bin_path}"'.format(src=src, bin_path=bin_path), "single-thread"),
    ]

    if show_progress:
        print("Compiling C++ BPE backend...")

    for cmd, mode in compile_cmds:
        t0 = time.time()
        code = os.system(cmd)
        dt = time.time() - t0
        if code == 0:
            if show_progress:
                print(f"C++ BPE compile finished in {dt:.2f}s (mode={mode})")
            return bin_path
        if show_progress:
            print(f"C++ BPE compile attempt failed (mode={mode}, exit={code}) after {dt:.2f}s")

    if show_progress:
        print("C++ BPE compile failed for all modes. Falling back to Python BPE.")
    return None


def _rebuild_token_to_bytes_from_merges(merges, eow_id=256):
    token_to_bytes = {i: [i] for i in range(256)}
    token_to_bytes[eow_id] = []
    for m in merges:
        a, b = m["pair"]
        nid = m["new_id"]
        token_to_bytes[nid] = token_to_bytes.get(a, []) + token_to_bytes.get(b, [])
    return token_to_bytes


def _train_byte_level_bpe_cpp(text, target_vocab_size, reserved_special, min_pair_freq, show_progress):
    bin_path = _compile_cpp_bpe(show_progress=show_progress)
    if bin_path is None:
        return None

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tf_in:
        tf_in.write(text)
        input_path = tf_in.name

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".json") as tf_out:
        output_path = tf_out.name

    try:
        cmd = (
            f'"{bin_path}" "{input_path}" '
            f'{int(target_vocab_size)} {int(reserved_special)} {int(min_pair_freq)} "{output_path}"'
        )
        if show_progress:
            print("Running C++ BPE merge trainer...")
        t0 = time.time()
        code = os.system(cmd)
        dt = time.time() - t0
        if code != 0:
            if show_progress:
                print(f"C++ BPE trainer failed (exit={code}) after {dt:.2f}s. Falling back to Python BPE.")
            return None
        if show_progress:
            print(f"C++ BPE merge trainer finished in {dt:.2f}s")

        with open(output_path, "r", encoding="utf-8") as f:
            merges = json.load(f)

        eow_id = 256
        token_to_bytes = _rebuild_token_to_bytes_from_merges(merges, eow_id=eow_id)
        next_id = 257 + len(merges)

        return {
            "merges": merges,
            "token_to_bytes": {str(k): v for k, v in token_to_bytes.items()},
            "vocab_size_without_specials": next_id,
            "eow_id": eow_id,
        }
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


def train_byte_level_bpe(text, target_vocab_size=50000, reserved_special=4, min_pair_freq=2, show_progress=True, prefer_cpp=True):
    if prefer_cpp and os.name == "posix":
        if show_progress:
            print("BPE backend preference: C++")
        cpp_result = _train_byte_level_bpe_cpp(
            text=text,
            target_vocab_size=target_vocab_size,
            reserved_special=reserved_special,
            min_pair_freq=min_pair_freq,
            show_progress=show_progress,
        )
        if cpp_result is not None:
            if show_progress:
                print("BPE backend used: C++")
            return cpp_result

    if show_progress:
        print("BPE backend used: Python")

    eow_id = 256
    base_vocab_size = 257
    max_bpe_tokens = max(base_vocab_size, target_vocab_size - reserved_special)

    pieces = _sentencepiece_like_pieces(text)
    word_counter = Counter(pieces)
    corpus_words = {}
    for piece, freq in word_counter.items():
        ids = tuple(list(piece.encode("utf-8", errors="ignore")) + [eow_id])
        corpus_words[ids] = corpus_words.get(ids, 0) + int(freq)

    merges = []
    token_to_bytes = {i: [i] for i in range(256)}
    token_to_bytes[eow_id] = []

    next_id = base_vocab_size

    total_merges = max_bpe_tokens - base_vocab_size
    pbar = tqdm(total=total_merges, desc="BPE merges", leave=False) if show_progress else None

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
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

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


def apply_bpe_to_piece(piece, merges_ranked, eow_id=256):
    return apply_bpe_to_word(piece, merges_ranked, eow_id=eow_id)


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
