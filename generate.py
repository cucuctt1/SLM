import torch

from config import Config
from model import GPTModel
from tokenizer import ByteLevelBPETokenizer
from checkpoint import auto_resume_latest
from quantization import load_quantized_checkpoint


def load_full_precision_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(ckpt["config"])

    tokenizer = ByteLevelBPETokenizer(vocab_size=cfg.vocab_size, context_length=cfg.context_length)
    tokenizer.set_state(ckpt["tokenizer_state"])

    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return cfg, tokenizer, model


def load_int4_model(quantized_checkpoint_path, device):
    payload = load_quantized_checkpoint(quantized_checkpoint_path, device=device)
    cfg = Config.from_dict(payload["config"])

    tokenizer = ByteLevelBPETokenizer(vocab_size=cfg.vocab_size, context_length=cfg.context_length)
    tokenizer.set_state(payload["tokenizer_state"])

    model = GPTModel(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    return cfg, tokenizer, model


def generate_text(model, tokenizer, prompt, device, max_new_tokens=120, temperature=0.9, top_k=40):
    chat_prompt = prompt if "Assistant:" in prompt else f"User: {prompt}\nAssistant:"
    context_length = getattr(model.cfg, "context_length", 512)
    encoded = tokenizer.encode(chat_prompt, add_special_tokens=True, max_length=10_000_000_000, padding=False, truncation=False)
    prompt_ids = encoded["input_ids"][-context_length:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_id,
            use_kv_cache=True,
        )

    return tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()

    quant_path = cfg.quantized_checkpoint_path
    if quant_path and torch.cuda.is_available() and __import__("os").path.exists(quant_path):
        _, tokenizer, model = load_int4_model(quant_path, device)
        print("Loaded int4 quantized model")
    else:
        latest = auto_resume_latest(cfg.checkpoint_dir)
        if latest is None:
            raise FileNotFoundError("No checkpoint found for generation.")
        _, tokenizer, model = load_full_precision_model(latest, device)
        print("Loaded full precision model")

    prompt = cfg.generate_prompt
    out = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=120,
        temperature=0.8,
        top_k=40,
    )
    print("\n=== GENERATED TEXT ===")
    print(out)


if __name__ == "__main__":
    main()
