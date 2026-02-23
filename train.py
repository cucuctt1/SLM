import math
import os
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from tokenizer import ByteLevelBPETokenizer
from dataset import (
    download_and_extract_kaggle_dataset,
    build_english_corpus,
    prepare_train_val_tokens,
    create_dataloaders,
)
from model import GPTModel
from checkpoint import save_checkpoint, load_checkpoint, auto_resume_latest
from quantization import save_quantized_checkpoint, load_quantized_checkpoint
from utils import (
    set_seed,
    count_parameters,
    estimate_model_memory_bytes,
    bytes_to_readable,
    gpu_memory_report,
    get_lr_cosine,
    ensure_dir,
    timestamp,
)


def evaluate(model, val_loader, device, max_batches=100):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                _, loss, _ = model(x, targets=y)
            losses.append(loss.item())
    model.train()
    if not losses:
        return float("inf"), float("inf")
    val_loss = float(np.mean(losses))
    ppl = math.exp(min(20.0, val_loss))
    return val_loss, ppl


def generate_sample(model, tokenizer, prompt, cfg, device):
    set_seed(cfg.seed)
    model.eval()
    encoded = tokenizer.encode(prompt, add_special_tokens=True, max_length=cfg.context_length, padding=False, truncation=True)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=cfg.generate_max_new_tokens,
            temperature=cfg.generate_temperature,
            top_k=cfg.generate_top_k,
            eos_token_id=tokenizer.eos_id,
            use_kv_cache=True,
        )
    text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    model.train()
    return text


def run_sanity_checks(cfg, model, tokenizer, optimizer, scheduler, scaler, device):
    print("\n=== SANITY CHECKS START ===")

    total_params, trainable_params = count_parameters(model)
    print(f"1) Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    target_low = 350_000_000
    target_high = 400_000_000
    in_range = target_low <= total_params <= target_high
    print(f"2) Parameter range check (350M-400M): {in_range}")

    tokenizer_vocab_size = len(tokenizer.vocab)
    print(f"3) Vocab size: tokenizer={tokenizer_vocab_size}, config={cfg.vocab_size}")

    print(f"4) Context length: {cfg.context_length}")

    x = torch.randint(0, cfg.vocab_size, (1, cfg.context_length), device=device)
    y = torch.randint(0, cfg.vocab_size, (1, cfg.context_length), device=device)

    try:
        logits, loss, _ = model(x, targets=y)
        print(f"5) Dummy forward pass OK | logits={tuple(logits.shape)} | loss={float(loss.item()):.6f}")
    except Exception as e:
        raise RuntimeError(f"Dummy forward pass failed: {e}")

    optimizer.zero_grad(set_to_none=True)
    try:
        scaler.scale(loss).backward()
        print("6) Single backward pass OK")
    except Exception as e:
        raise RuntimeError(f"Single backward pass failed: {e}")

    grad_nonzero = False
    for p in model.parameters():
        if p.grad is not None and torch.any(torch.abs(p.grad) > 0):
            grad_nonzero = True
            break
    if not grad_nonzero:
        raise RuntimeError("7) Gradient update check failed: all grads are zero or None")
    print("7) Gradient update check passed")

    mem_bytes = estimate_model_memory_bytes(total_params)
    print(f"8) Estimated training memory (rough): {bytes_to_readable(mem_bytes)}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    print("9) 1 mini train step completed")

    prompt_ids = tokenizer.encode("Hello world", add_special_tokens=True, max_length=32, padding=False, truncation=True)["input_ids"]
    input_ids = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=16,
            temperature=1.0,
            top_k=20,
            eos_token_id=tokenizer.eos_id,
            use_kv_cache=True,
        )
    _ = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    print("10) Inference generation check passed")

    q_tmp = os.path.join(cfg.checkpoint_dir, "_sanity_quantized.pt")
    save_quantized_checkpoint(q_tmp, model, cfg, tokenizer.get_state())
    q_payload = load_quantized_checkpoint(q_tmp, device=device)

    q_model = GPTModel(cfg).to(device)
    q_model.load_state_dict(q_payload["state_dict"], strict=True)
    q_model.eval()

    with torch.no_grad():
        _ = q_model.generate(
            input_ids=input_ids,
            max_new_tokens=8,
            temperature=1.0,
            top_k=10,
            eos_token_id=tokenizer.eos_id,
            use_kv_cache=True,
        )
    print("11) Quantized inference test passed")

    try:
        _ = q_model(input_ids)
        print("12) Shape mismatch check passed")
    except Exception as e:
        raise RuntimeError(f"12) Shape mismatch check failed: {e}")

    print("GPU memory:", gpu_memory_report())
    print("=== SANITY CHECKS END ===\n")


def main():
    cfg = Config()
    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.tokenizer_dir)
    ensure_dir(os.path.dirname(cfg.corpus_path))

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[{timestamp()}] Device: {device}")
    print(f"[{timestamp()}] Config param estimate (12*L*d^2): {cfg.target_param_estimate:,}")

    print(f"[{timestamp()}] Downloading and extracting Kaggle dataset...")
    extract_dir = download_and_extract_kaggle_dataset(
        dataset_slug=cfg.dataset_slug,
        download_dir=cfg.dataset_download_dir,
        extract_dir=cfg.dataset_extract_dir,
        force=False,
    )

    print(f"[{timestamp()}] Building merged English corpus...")
    corpus_path, num_lines, num_chars = build_english_corpus(
        extract_dir=extract_dir,
        corpus_path=cfg.corpus_path,
        max_chars=cfg.max_corpus_chars,
    )
    print(f"Corpus lines={num_lines:,}, chars={num_chars:,}, path={corpus_path}")

    tokenizer = ByteLevelBPETokenizer(vocab_size=cfg.vocab_size, context_length=cfg.context_length)
    if tokenizer.try_load(cfg.tokenizer_dir):
        print(f"[{timestamp()}] Loading existing tokenizer...")
    else:
        print(f"[{timestamp()}] Training byte-level BPE tokenizer to vocab={cfg.vocab_size}...")
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            corpus_text = f.read(getattr(cfg, "tokenizer_train_chars", 20_000_000))
        tokenizer.train(corpus_text)
        tokenizer.save(cfg.tokenizer_dir)

    print(f"Tokenizer vocab size actual: {len(tokenizer.vocab)}")

    print(f"[{timestamp()}] Preparing tokenized train/val splits (90/10)...")
    train_count, val_count = prepare_train_val_tokens(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        train_tokens_path=cfg.train_tokens_path,
        val_tokens_path=cfg.val_tokens_path,
        train_split=cfg.train_split,
    )
    print(f"Train tokens={train_count:,}, Val tokens={val_count:,}")

    train_loader, val_loader = create_dataloaders(
        train_tokens_path=cfg.train_tokens_path,
        val_tokens_path=cfg.val_tokens_path,
        context_length=cfg.context_length,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
    )

    model = GPTModel(cfg).to(device)

    total_params, _ = count_parameters(model)
    print(f"Actual parameter count: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    total_train_steps = (len(train_loader) * cfg.epochs) // max(1, cfg.grad_accum_steps)
    total_train_steps = max(1, total_train_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_cosine(
            step=step,
            total_steps=total_train_steps,
            warmup_steps=cfg.warmup_steps,
            max_lr=1.0,
            min_lr=cfg.min_lr / max(cfg.learning_rate, 1e-12),
        ),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and torch.cuda.is_available()))

    start_epoch = 0
    global_step = 0

    latest = auto_resume_latest(cfg.checkpoint_dir)
    if latest is not None:
        print(f"[{timestamp()}] Resuming from checkpoint: {latest}")
        info = load_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = info["epoch"] + 1
        global_step = info["global_step"]
        if info.get("tokenizer_state") is not None:
            tokenizer.set_state(info["tokenizer_state"])

    run_sanity_checks(cfg, model, tokenizer, optimizer, scheduler, scaler, device)

    print(f"[{timestamp()}] Starting training...")
    model.train()

    for epoch in range(start_epoch, cfg.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.epochs}")
        running_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for step_idx, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and torch.cuda.is_available())):
                _, loss, _ = model(x, targets=y)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg.grad_accum_steps

            if (step_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / (step_idx + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}", mem=gpu_memory_report())

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    break

        train_loss_epoch = running_loss / max(1, len(train_loader))
        val_loss, ppl = evaluate(model, val_loader, device)

        print(
            f"[{timestamp()}] Epoch={epoch+1} train_loss={train_loss_epoch:.4f} "
            f"val_loss={val_loss:.4f} perplexity={ppl:.2f} gpu_mem=({gpu_memory_report()})"
        )

        save_path = save_checkpoint(
            checkpoint_dir=cfg.checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            cfg=cfg,
            tokenizer_state=tokenizer.get_state(),
        )
        print(f"Saved checkpoint: {save_path}")

        if (epoch + 1) % cfg.eval_every_epochs == 0:
            sample = generate_sample(model, tokenizer, cfg.generate_prompt, cfg, device)
            print("\n===== SAMPLE GENERATION =====")
            print(sample)
            print("===== END SAMPLE =====\n")

        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            print("Reached max_steps. Stopping early.")
            break

    print(f"[{timestamp()}] Training complete.")
    q_path = save_quantized_checkpoint(
        path=cfg.quantized_checkpoint_path,
        model=model,
        cfg=cfg,
        tokenizer_state=tokenizer.get_state(),
    )
    print(f"Quantized checkpoint saved: {q_path}")

    q_payload = load_quantized_checkpoint(q_path, device=device)
    q_model = GPTModel(cfg).to(device)
    q_model.load_state_dict(q_payload["state_dict"], strict=True)
    q_model.eval()

    prompt_ids = tokenizer.encode(cfg.generate_prompt, add_special_tokens=True, max_length=64, padding=False, truncation=True)["input_ids"]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = q_model.generate(
            input_ids=input_ids,
            max_new_tokens=48,
            temperature=0.9,
            top_k=40,
            eos_token_id=tokenizer.eos_id,
            use_kv_cache=True,
        )
    print("Quantized generation test output:")
    print(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
