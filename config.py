import os


class Config:
    def __init__(self):
        self.seed = 42

        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or os.environ.get("COLAB_GPU") else ("cuda" if __import__("torch").cuda.is_available() else "cpu")


        self.vocab_size = 50_000
        self.context_length = 512

        self.dropout = 0.1
        self.bias = True

        self.n_layers = 16
        self.d_model = 768
        self.n_heads = 12
        self.d_ff = 3072

        self.batch_size = 16
        self.grad_accum_steps = 2
        self.eval_batch_size = 2
        self.epochs = 8

        self.learning_rate = 2e-4
        self.min_lr = 2e-5
        self.weight_decay = 0.1
        self.betas = (0.9, 0.95)
        self.grad_clip = 1.0
        self.warmup_steps = 500

        self.use_amp = True
        self.grad_checkpointing = False

        self.dataset_source = "huggingface"
        self.dataset_slug = "HuggingFaceH4/ultrachat_200k"
        self.hf_config_name = "default"
        self.hf_split_candidates = ["train_sft", "train", "train_gen", "test_sft", "test"]
        self.hf_page_size = 1000
        self.hf_request_sleep = 0.20
        self.hf_resume_stream = True
        self.hf_force_rebuild_corpus = False
        self.dataset_download_dir = "/content/data"
        self.dataset_extract_dir = "/content/data/extracted"
        self.corpus_path = "/content/data/corpus.txt"
        self.train_tokens_path = "/content/data/train_tokens.npy"
        self.val_tokens_path = "/content/data/val_tokens.npy"
        self.train_split = 0.9
        self.max_corpus_chars = 60_000_000
        self.tokenizer_train_chars = 20_000_000
        self.use_sentencepiece_package = True
        self.force_retrain_tokenizer = False
        self.force_retokenize = False

        self.checkpoint_dir = "/content/checkpoints"
        self.tokenizer_dir = "/content/tokenizer"
        self.tokenizer_pack_path = "/content/tokenizer/tokenizer.pack.json"

        self.eval_every_epochs = 1
        self.generate_prompt = "User: Explain what overfitting is in simple terms.\nAssistant:"
        self.generate_max_new_tokens = 120
        self.generate_temperature = 0.9
        self.generate_top_k = 40

        self.quantized_checkpoint_path = "/content/checkpoints/quantized_int4.pt"

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def target_param_estimate(self):
        return int(12 * self.n_layers * (self.d_model ** 2))

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        cfg = cls()
        for k, v in d.items():
            setattr(cfg, k, v)
        return cfg
