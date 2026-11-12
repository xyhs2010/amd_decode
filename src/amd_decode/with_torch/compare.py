"""Compare dense prefill against token-by-token decode.

The cache-format path is deliberately separate: this script's first job is to
prove that the decode math (RoPE, GQA, KV append, attention) matches a dense
forward of the same tokens. Once that is green, the saved hydra_llm cache can
be used as the starting state for a real decode.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Qwen3Config
from .kv_cache_file import SavedKVCache, SavedKVCacheView
from .model import KVCache, Qwen3Torch
from .rope import build_cos_sin


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    # Cosine over ~150k logits is inaccurate in fp32 (norms are ~5e3); use
    # fp64 so a bitwise-identical pair reports exactly 1.0.
    cosine = F.cosine_similarity(a.double(), b.double(), dim=-1).mean().item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "cosine": cosine,
    }


def _fmt_tokens(tokenizer, ids) -> str:
    if tokenizer is None:
        return " ".join(str(int(i)) for i in ids)
    return repr(tokenizer.decode(list(int(i) for i in ids)))


def _load_tokenizer(model_dir: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:  # noqa: BLE001 - optional convenience only
        print(f"[info] tokenizer unavailable ({type(exc).__name__}: {exc})")
        return None


def _get_tokens(args, tokenizer) -> torch.Tensor:
    if args.sample:
        try:
            cache = SavedKVCache(args.sample)
            return cache.tokens()[: args.tokens]
        except (PermissionError, FileNotFoundError, OSError) as exc:
            print(f"[warn] cannot read --sample ({exc}); falling back to synthetic tokens")
    if tokenizer is not None:
        text = "The capital of France is"
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: args.tokens]
        if len(ids) < args.tokens:
            ids = (ids * ((args.tokens // max(len(ids), 1)) + 1))[: args.tokens]
        return torch.tensor(ids, dtype=torch.long)
    return torch.arange(1000, 1000 + args.tokens, dtype=torch.long)


def run_selfcheck(model: Qwen3Torch, tokens: torch.Tensor, tokenizer) -> None:
    n = tokens.numel()
    print(f"\n== prefill vs decode self-check ({n} tokens) ==")
    print(f"tokens: {_fmt_tokens(tokenizer, tokens.tolist())}\n")

    t0 = time.perf_counter()
    logits_prefill = model.forward(tokens, kv=None, position_offset=0)[0].float().cpu()
    _sync(model.device)
    t_prefill = time.perf_counter() - t0

    kv = KVCache(model.num_layers)
    decode_logits: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for i in range(n):
        out = model.forward(tokens[i : i + 1], kv=kv, position_offset=i)
        decode_logits.append(out[0, -1].float().cpu())
    _sync(model.device)
    t_decode = time.perf_counter() - t0

    print(f"{'pos':>4} {'token':>7} {'max|d|':>10} {'mean|d|':>10} {'cosine':>10} "
          f"{'top1':>6} {'prefill_argmax':>15} {'decode_argmax':>14}")
    all_top1 = True
    for i in range(n):
        a = logits_prefill[i]
        b = decode_logits[i]
        s = _stats(a, b)
        top1 = int(a.argmax()) == int(b.argmax())
        all_top1 &= top1
        print(
            f"{i:>4} {int(tokens[i]):>7} {s['max_abs']:>10.5f} {s['mean_abs']:>10.6f} "
            f"{s['cosine']:>10.7f} {str(top1):>6} {int(a.argmax()):>15} {int(b.argmax()):>14}"
        )

    s = _stats(logits_prefill, torch.stack(decode_logits))
    print(
        f"\noverall: max|d|={s['max_abs']:.5f} mean|d|={s['mean_abs']:.6f} "
        f"cosine={s['cosine']:.8f} top1_match={all_top1}"
    )

    last = logits_prefill[-1]
    top = torch.topk(last, 5)
    print(f"\nprefill top-5 at last position: {top.indices.tolist()}  "
          f"logits={[round(float(v), 4) for v in top.values]}")
    if tokenizer is not None:
        print("  tokens:", [repr(tokenizer.decode([int(i)])) for i in top.indices])
    print(f"\ntiming: prefill {t_prefill:.2f}s, decode {t_decode:.2f}s "
          f"({t_decode / max(n, 1):.3f}s/token)")
    if model.device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def run_kv_compare(model: Qwen3Torch, sample: str, n_tokens: int = 512) -> None:
    """Compare locally computed layer-0 K/V with the saved cache.

    This checks three things at once: the cache stores post-RoPE K, our YaRN
    matches the prefill's, and the q8 block-scale dequantization is laid out
    correctly. With int4 weights the only remaining difference should be
    bf16-vs-fp16 activation arithmetic.
    """
    print("\n== local layer-0 K/V vs saved cache ==")
    try:
        cache = SavedKVCache(sample)
        tokens = cache.tokens()[:n_tokens]
    except (PermissionError, OSError) as exc:
        print(f"[warn] cannot read cache: {exc}")
        return

    n = tokens.numel()
    p = model._load_layer(0)
    x = model.embed[tokens].unsqueeze(0).to(model.device).to(model.dtype)
    cos, sin = build_cos_sin(
        torch.arange(n, device=model.device),
        model.inv_freq,
        model.attention_scaling,
        model.dtype,
    )
    _, k, v = model._attn_prep(model._rms(x, p["input_layernorm.weight"]), p, cos, sin)
    k_local = k[0].float().cpu()
    v_local = v[0].float().cpu()
    del p, x, k, v

    k_saved, v_saved = cache.layer(0, device="cpu", dtype=torch.float32)
    k_saved = k_saved[0, :, :n].float()
    v_saved = v_saved[0, :, :n].float()

    def report(name: str, a: torch.Tensor, b: torch.Tensor, norm_ratio: bool = False) -> None:
        c = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1)
        print(
            f"{name}: cos mean={c.mean():.6f} min={c.min():.6f} p1={torch.quantile(c, 0.01):.6f} | "
            f"mean|x| local={a.abs().mean():.5f} saved={b.abs().mean():.5f} | "
            f"max|x| local={a.abs().max():.3f} saved={b.abs().max():.3f}"
        )
        if norm_ratio:
            r = a.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)
            print(
                f"    per-head norm ratio: mean={r.mean():.5f} std={r.std():.5f} "
                f"min={r.min():.3f} max={r.max():.3f}"
            )

    print(f"tokens={n} weight_bits={model.weight_bits} dtype={model.dtype}")
    report("K", k_local, k_saved, norm_ratio=True)
    report("V", v_local, v_saved)


def run_cache_smoke(model: Qwen3Torch, sample: str, tokenizer) -> None:
    """Decode one token from a saved cache.

    The saved cache covers exactly ``tokens``; there is no held-out next token,
    so this is a smoke test of the dequantization + decode path, not a quality
    measurement.
    """
    print(f"\n== saved-cache smoke test ==\n{sample}")
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        cache = SavedKVCache(sample)
        tokens = cache.tokens()
    except (PermissionError, OSError) as exc:
        print(f"[warn] cannot read cache: {exc}")
        return

    print(
        f"meta: seq_len={cache.seq_len} quant={cache.meta.get('quant')} "
        f"dtype={cache.meta.get('dtype')} block_size={cache.block_size} "
        f"scheme={cache.meta.get('scheme')}"
    )
    try:
        k, v = cache.layer(0, device=model.device, dtype=model.dtype)
    except PermissionError as exc:
        print(f"[warn] cannot read layer files: {exc}")
        return
    print(
        f"layer 0 KV: k{tuple(k.shape)} {k.dtype} mean|k|={k.float().abs().mean():.5f} "
        f"max|k|={k.float().abs().max():.3f}; "
        f"v{tuple(v.shape)} mean|v|={v.float().abs().mean():.5f} max|v|={v.float().abs().max():.3f}"
    )
    del k, v

    view = SavedKVCacheView(cache, device=model.device, dtype=model.dtype)
    next_id = int(tokens[-1])
    t0 = time.perf_counter()
    logits = model.forward(torch.tensor([[next_id]]), kv=view, position_offset=cache.seq_len)
    _sync(model.device)
    elapsed = time.perf_counter() - t0
    logits = logits[0, -1].float()
    top = torch.topk(logits, 5)
    print(f"decoded at position {cache.seq_len} (input id {next_id}) in {elapsed:.2f}s")
    print(f"top-5: {top.indices.tolist()} logits={[round(float(v), 4) for v in top.values]}")
    if tokenizer is not None:
        print("  tokens:", [repr(tokenizer.decode([int(i)])) for i in top.indices])
    if model.device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="data/models/Qwen3-8B-Base")
    ap.add_argument("--sample", default=None, help="KV cache sample dir (for tokens / cache smoke test)")
    ap.add_argument("--tokens", type=int, default=8, help="number of tokens for prefill-vs-decode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    ap.add_argument("--weight-bits", type=int, default=4, choices=(4, 16),
                    help="4 = torchao-tinygemm-compatible int4, 16 = raw bf16 weights")
    ap.add_argument("--cache-decode", action="store_true", help="also run one decode step from --sample")
    ap.add_argument("--compare-kv", action="store_true", help="compare local layer-0 K/V against --sample")
    ap.add_argument("--kv-tokens", type=int, default=512, help="tokens for --compare-kv")
    ap.add_argument("--skip-selfcheck", action="store_true", help="skip the prefill-vs-decode check")
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"model dir not found: {args.model}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device requested but torch.cuda.is_available() is False")

    cfg = Qwen3Config.from_model_dir(args.model)
    print(
        f"model={args.model}\n"
        f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
        f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
        f"head_dim={cfg.head_dim} vocab={cfg.vocab_size}\n"
        f"rope={cfg.rope_scaling} theta={cfg.rope_theta}\n"
        f"device={device} dtype={args.dtype} weight_bits={args.weight_bits}"
    )

    tokenizer = None if args.no_tokenizer else _load_tokenizer(args.model)
    tokens = _get_tokens(args, tokenizer)
    dtype = getattr(torch, args.dtype)

    model = Qwen3Torch(args.model, device=device, dtype=dtype, weight_bits=args.weight_bits)
    if not args.skip_selfcheck:
        run_selfcheck(model, tokens, tokenizer)
    if args.compare_kv:
        if not args.sample:
            print("[warn] --compare-kv needs --sample")
        else:
            run_kv_compare(model, args.sample, args.kv_tokens)
    if args.cache_decode:
        if not args.sample:
            print("[warn] --cache-decode needs --sample")
        else:
            run_cache_smoke(model, args.sample, tokenizer)


if __name__ == "__main__":
    main()
