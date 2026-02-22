import os
import torch


def _to_int4_signed(q_int8):
    q = q_int8.to(torch.int16)
    q = torch.where(q < 0, q + 16, q)
    return q.to(torch.uint8)


def _from_int4_signed(q_u8):
    q = q_u8.to(torch.int16)
    q = torch.where(q >= 8, q - 16, q)
    return q.to(torch.int8)


def pack_int4(q_int8_flat):
    q_u8 = _to_int4_signed(q_int8_flat)
    if q_u8.numel() % 2 != 0:
        q_u8 = torch.cat([q_u8, torch.zeros(1, dtype=torch.uint8, device=q_u8.device)], dim=0)
    low = q_u8[0::2] & 0x0F
    high = (q_u8[1::2] & 0x0F) << 4
    packed = (low | high).contiguous()
    return packed


def unpack_int4(packed_u8, original_numel):
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    out = torch.empty(low.numel() * 2, dtype=torch.uint8, device=packed_u8.device)
    out[0::2] = low
    out[1::2] = high
    out = out[:original_numel]
    q_int8 = _from_int4_signed(out)
    return q_int8


def quantize_tensor_int4(tensor):
    if tensor.numel() == 0:
        return {
            "packed": torch.empty(0, dtype=torch.uint8),
            "shape": list(tensor.shape),
            "scale": 1.0,
            "zero_point": 0,
            "numel": 0,
            "dtype": str(tensor.dtype),
        }

    t = tensor.detach().to(torch.float32).cpu().contiguous().view(-1)
    max_abs = torch.max(torch.abs(t)).item()
    scale = max(max_abs / 7.0, 1e-8)
    zero_point = 0

    q = torch.round(t / scale).clamp(-8, 7).to(torch.int8)
    packed = pack_int4(q)

    return {
        "packed": packed,
        "shape": list(tensor.shape),
        "scale": float(scale),
        "zero_point": int(zero_point),
        "numel": int(t.numel()),
        "dtype": str(tensor.dtype),
    }


def dequantize_tensor_int4(q_state, device="cpu"):
    packed = q_state["packed"].to(device)
    q_int8 = unpack_int4(packed, q_state["numel"]).to(torch.float32)
    t = (q_int8 - q_state["zero_point"]) * q_state["scale"]
    shape = tuple(q_state["shape"])
    return t.view(shape)


def quantize_model_state_dict(state_dict):
    q_state = {}
    for name, tensor in state_dict.items():
        q_state[name] = quantize_tensor_int4(tensor)
    return q_state


def dequantize_state_dict_on_the_fly(q_state_dict, device="cpu"):
    out = {}
    for name, q_state in q_state_dict.items():
        out[name] = dequantize_tensor_int4(q_state, device=device)
    return out


def save_quantized_checkpoint(path, model, cfg, tokenizer_state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    q_state = quantize_model_state_dict(model.state_dict())
    payload = {
        "format": "int4_symmetric_per_tensor",
        "model_state_int4": q_state,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        "tokenizer_state": tokenizer_state,
    }
    torch.save(payload, path)
    return path


def load_quantized_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu")
    q_state = payload["model_state_int4"]
    deq_state = dequantize_state_dict_on_the_fly(q_state, device=device)
    return {
        "state_dict": deq_state,
        "config": payload["config"],
        "tokenizer_state": payload["tokenizer_state"],
        "format": payload.get("format", "int4_symmetric_per_tensor"),
    }
