import os
import torch


def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    cfg,
    tokenizer_state,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_step_{global_step}.pt")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        "tokenizer_state": tokenizer_state,
    }

    torch.save(ckpt, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "config": ckpt.get("config"),
        "tokenizer_state": ckpt.get("tokenizer_state"),
    }


def auto_resume_latest(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
    if not files:
        return None

    def step_of(name):
        try:
            return int(name.replace("ckpt_step_", "").replace(".pt", ""))
        except Exception:
            return -1

    files.sort(key=step_of)
    latest = files[-1]
    return os.path.join(checkpoint_dir, latest)
