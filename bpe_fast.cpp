#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct VecHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 1469598103934665603ull;
        for (int x : v) {
            h ^= static_cast<std::size_t>(x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
            h *= 1099511628211ull;
        }
        return h;
    }
};

static inline uint64_t pair_key(uint32_t a, uint32_t b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

static inline std::pair<int, int> decode_pair_key(uint64_t key) {
    int a = static_cast<int>(key >> 32);
    int b = static_cast<int>(key & 0xffffffffu);
    return {a, b};
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: bpe_fast <input_text_path> <target_vocab_size> <reserved_special> <min_pair_freq> <output_merges_json>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const int target_vocab_size = std::stoi(argv[2]);
    const int reserved_special = std::stoi(argv[3]);
    const int min_pair_freq = std::stoi(argv[4]);
    const std::string output_path = argv[5];

    const int eow_id = 256;
    const int base_vocab_size = 257;
    const int max_bpe_tokens = std::max(base_vocab_size, target_vocab_size - reserved_special);

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input: " << input_path << "\n";
        return 2;
    }

    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    std::unordered_map<std::string, long long> word_freq;
    std::string cur;
    cur.reserve(64);

    for (unsigned char c : text) {
        if (std::isspace(c)) {
            if (!cur.empty()) {
                word_freq[cur] += 1;
                cur.clear();
            }
        } else {
            cur.push_back(static_cast<char>(c));
        }
    }
    if (!cur.empty()) {
        word_freq[cur] += 1;
    }

    std::unordered_map<std::vector<int>, long long, VecHash> corpus;
    corpus.reserve(word_freq.size() * 2 + 1);

    for (const auto& kv : word_freq) {
        const std::string& w = kv.first;
        long long freq = kv.second;
        std::vector<int> ids;
        ids.reserve(w.size() + 1);
        for (unsigned char c : w) {
            ids.push_back(static_cast<int>(c));
        }
        ids.push_back(eow_id);
        corpus[ids] += freq;
    }

    struct MergeRec {
        int a;
        int b;
        int new_id;
        long long freq;
    };

    std::vector<MergeRec> merges;
    merges.reserve(std::max(0, max_bpe_tokens - base_vocab_size));

    int next_id = base_vocab_size;
    const int total_target = std::max(0, max_bpe_tokens - base_vocab_size);

    while (next_id < max_bpe_tokens) {
        std::unordered_map<uint64_t, long long> stats;
        stats.reserve(corpus.size() * 2 + 1);

        for (const auto& kv : corpus) {
            const std::vector<int>& seq = kv.first;
            long long freq = kv.second;
            if (seq.size() < 2) {
                continue;
            }
            for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
                uint64_t k = pair_key(static_cast<uint32_t>(seq[i]), static_cast<uint32_t>(seq[i + 1]));
                stats[k] += freq;
            }
        }

        if (stats.empty()) {
            break;
        }

        uint64_t best_key = 0;
        long long best_freq = -1;
        for (const auto& kv : stats) {
            if (kv.second > best_freq) {
                best_freq = kv.second;
                best_key = kv.first;
            }
        }

        if (best_freq < min_pair_freq) {
            break;
        }

        auto best_pair = decode_pair_key(best_key);
        int a = best_pair.first;
        int b = best_pair.second;

        std::unordered_map<std::vector<int>, long long, VecHash> new_corpus;
        new_corpus.reserve(corpus.size() * 2 + 1);

        for (const auto& kv : corpus) {
            const std::vector<int>& seq = kv.first;
            long long freq = kv.second;

            if (seq.size() < 2) {
                new_corpus[seq] += freq;
                continue;
            }

            std::vector<int> merged;
            merged.reserve(seq.size());

            std::size_t i = 0;
            while (i < seq.size()) {
                if (i + 1 < seq.size() && seq[i] == a && seq[i + 1] == b) {
                    merged.push_back(next_id);
                    i += 2;
                } else {
                    merged.push_back(seq[i]);
                    i += 1;
                }
            }

            new_corpus[merged] += freq;
        }

        corpus.swap(new_corpus);

        merges.push_back(MergeRec{a, b, next_id, best_freq});
        ++next_id;

        int done = static_cast<int>(merges.size());
        if (done % 25 == 0 || done == total_target) {
            std::cerr << "[BPE C++] merges: " << done << "/" << total_target << "\r" << std::flush;
        }
    }

    std::cerr << "[BPE C++] merges: " << merges.size() << "/" << total_target << "\n";

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output: " << output_path << "\n";
        return 3;
    }

    out << "[";
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const auto& m = merges[i];
        out << "{\"pair\":[" << m.a << "," << m.b << "],\"new_id\":" << m.new_id << ",\"freq\":" << m.freq << "}";
        if (i + 1 < merges.size()) {
            out << ",";
        }
    }
    out << "]";
    out.close();

    return 0;
}
