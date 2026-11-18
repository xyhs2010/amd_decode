"""Dense prefill for Qwen3.5-9B, saved as the hybrid decode cache.

Run this on a CUDA machine that can hold the bf16 model (a 32 GB 5090 is
enough with q8 K/V and a small chunk size).  It is deliberately independent of
``amd_decode``: only transformers and safetensors are needed.

What is saved per sample (mirroring the existing hydra_llm layout)::

    <out>/<model_tag>/<quant>/<benchmark>/L<length>/<task>/<sample_id>/
        meta.json
        tokens.safetensors                 # int32 token ids of the prompt
        layer_00.safetensors               # linear layer:
            conv_state      [1, 8192, 4]       fp16
            recurrent_state [1, 32, 128, 128]  fp32
        layer_03.safetensors               # full-attention layer, q8:
            k_q     int8    [1, 4, T, 256]
            k_scale float32 [1, 4, n_blocks, 1, 256]   # block = 8192
            v_q     int8    [1, 4, T, 256]
            v_scale float32 [1, 4, T, 1]

The 24 linear layers store the DeltaNet conv/recurrent state; the 8 full
attention layers store K/V.  The answer prefix is *not* part of the prefill,
exactly like the RULER caches.

Examples::

    python prefill/prefill_qwen35.py --model data/models/Qwen3.5-9B \
        --benchmark ruler --lengths 16384 --tasks niah_single_2 cwe \
        --limit 2 --out data/kv_cache

    # generic JSONL: one object per line with
    #   {"sample_id", "task", "length", "prompt"}
    python prefill/prefill_qwen35.py --jsonl longbench_v2.jsonl --chunk-size 4096
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as exc:  # pragma: no cover - depends on the remote env
    raise SystemExit(
        "this script needs transformers with Qwen3.5 support (>= 5.23)"
    ) from exc


# --------------------------------------------------------------------------- ruler / jsonl


def load_ruler(root: Path, lengths, tasks, limit: int):
    for length in lengths:
        for task in tasks:
            path = root / f"L{length}" / task / "test.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing RULER file: {path}")
            kept = 0
            for line in path.open():
                row = json.loads(line)
                yield {
                    "sample_id": f"ruler_{task}_{length}_{row['index']}",
                    "task": task,
                    "length": length,
                    "prompt": row["input"],
                }
                kept += 1
                if limit and kept >= limit:
                    break


def load_jsonl(path: Path, limit: int):
    for line in path.open():
        row = json.loads(line)
        yield {
            "sample_id": str(row["sample_id"]),
            "task": str(row.get("task", path.stem)),
            "length": int(row.get("length", 0)),
            "prompt": str(row.get("prompt", row.get("context"))),
        }
        if limit:
            limit -= 1
            if limit == 0:
                break


# --------------------------------------------------------------------------- quantized cache


def _quant_k(k: torch.Tensor, block: int):
    """[B,H,T,D] -> int8 q + float32 per-(head, block, channel) scale."""
    b, h, t, d = k.shape
    n_blocks = -(-t // block)
    pad = n_blocks * block - t
    if pad:
        k = torch.cat((k, k[..., -1:, :].expand(b, h, pad, d)), dim=-2)
    blocks = k.view(b, h, n_blocks, block, d)
    scale = (blocks.abs().amax(dim=3, keepdim=True) / 127.0).clamp_min(1e-8)
    q = torch.round(blocks / scale).clamp(-127, 127).to(torch.int8)
    return q.view(b, h, n_blocks * block, d)[..., :t, :], scale.float()


def _quant_v(v: torch.Tensor):
    """[B,H,T,D] -> int8 q + float32 per-token scale."""
    scale = (v.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-8)
    q = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def _dequant_k(q: torch.Tensor, scale: torch.Tensor, block: int):
    b, h, t, d = q.shape
    n_blocks = scale.shape[2]
    pad = n_blocks * block - t
    q = F.pad(q, (0, 0, 0, pad)) if pad else q
    q = q.view(b, h, n_blocks, block, d).float()
    out = (q * scale).view(b, h, n_blocks * block, d)
    return out[..., :t, :]


class HybridQuantCache(DynamicCache):
    """DynamicCache with the full-attention K/V kept as int8 on the GPU.

    The linear-attention layers are the stock ``LinearAttentionLayer`` from
    transformers; only the attention ``update`` path is replaced.  K is
    quantized per (head, block, channel) and V per token, which is the same
    scheme the decode backend seeds from.
    """

    def __init__(self, config, scheme: str = "q8", block: int = 8192):
        super().__init__(config=config)
        self.scheme = scheme
        self.block = block
        self._store: dict[int, dict[str, torch.Tensor]] = {}
        self._seq_len = 0

    @property
    def is_compileable(self) -> bool:
        return False

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    def get_mask_sizes(self, q_length: int, layer_idx: int = 0):
        return self._seq_len + q_length, 0

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        old = self._store.get(layer_idx)
        if old is not None:
            if self.scheme == "fp16":
                k_full = torch.cat((old["k"].to(key_states.dtype), key_states), dim=-2)
                v_full = torch.cat((old["v"].to(value_states.dtype), value_states), dim=-2)
            else:
                k_full = torch.cat(
                    (_dequant_k(old["k_q"], old["k_scale"], self.block).to(key_states.dtype),
                     key_states), dim=-2,
                )
                v_full = torch.cat(
                    ((old["v_q"] * old["v_scale"]).to(value_states.dtype), value_states),
                    dim=-2,
                )
        else:
            k_full, v_full = key_states, value_states

        if self.scheme == "fp16":
            self._store[layer_idx] = {
                "k": k_full.to(torch.float16),
                "v": v_full.to(torch.float16),
            }
        else:
            k_q, k_scale = _quant_k(k_full, self.block)
            v_q, v_scale = _quant_v(v_full)
            self._store[layer_idx] = {
                "k_q": k_q, "k_scale": k_scale, "v_q": v_q, "v_scale": v_scale,
            }
        self._seq_len = k_full.shape[-2]
        return k_full, v_full

    def snapshot(self, layer_types) -> dict[int, dict[str, torch.Tensor]]:
        """Move every layer's cache to CPU, ready for a writer thread."""
        out: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                out[layer_idx] = {
                    name: t.detach().to("cpu") for name, t in self._store[layer_idx].items()
                }
            else:
                layer = self.layers[layer_idx]
                out[layer_idx] = {
                    "conv_state": layer.conv_states[0].detach().to("cpu", torch.float16),
                    "recurrent_state": layer.recurrent_states[0].detach().to("cpu"),
                }
        return out


# --------------------------------------------------------------------------- prefill / io


@torch.inference_mode()
def dense_prefill(model, ids: torch.Tensor, cache, chunk: int) -> None:
    n = ids.shape[1]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        kwargs = dict(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True)
        try:
            model(**kwargs, logits_to_keep=1)
        except TypeError:
            model(**kwargs)


def sample_path(root: Path, tag: str, quant: str, benchmark: str, length: int,
                task: str, sample_id: str) -> Path:
    return root / tag / quant / benchmark / f"L{length}" / task / sample_id


def is_complete(path: Path, num_layers: int) -> bool:
    return (path / "meta.json").exists() and all(
        (path / f"layer_{i:02d}.safetensors").exists() for i in range(num_layers)
    )


def save_sample(path: Path, layers, meta: dict, tokens: list[int]) -> None:
    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if path.exists():
        shutil.rmtree(path)
    save_file({"tokens": torch.tensor(tokens, dtype=torch.int32)},
              str(tmp / "tokens.safetensors"))
    for layer_idx, tensors in layers.items():
        save_file({name: t.contiguous() for name, t in tensors.items()},
                  str(tmp / f"layer_{layer_idx:02d}.safetensors"))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(path)


@torch.inference_mode()
def verify(model, cfg, ids: list[int], scheme: str, block: int) -> None:
    """Compare the quantized cache's next-token logits against fp16."""
    if len(ids) < 2:
        return
    prefix = ids[: min(len(ids) - 1, 4096)]
    last = ids[len(prefix)]
    ids_t = torch.tensor([prefix], dtype=torch.long, device="cuda")
    caches = {
        "fp16": HybridQuantCache(cfg, "fp16", block),
        scheme: HybridQuantCache(cfg, scheme, block),
    }
    for cache in caches.values():
        dense_prefill(model, ids_t, cache, chunk=1024)

    out = {}
    for name, cache in caches.items():
        logits = model(
            input_ids=torch.tensor([[last]], dtype=torch.long, device="cuda"),
            past_key_values=cache, use_cache=True,
        ).logits[:, -1, :].float()
        out[name] = logits
    ref, got = out["fp16"], out[scheme]
    diff = (ref - got).abs()
    print(
        f"verify ({scheme}, prefix={len(prefix)}): "
        f"max|dlogit|={diff.max().item():.4f} "
        f"mean|dlogit|={diff.mean().item():.4f} "
        f"top1_match={bool(ref.argmax(-1) == got.argmax(-1))}"
    )
    del caches, out, ids_t
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="data/models/Qwen3.5-9B")
    ap.add_argument("--out", default="data/kv_cache")
    ap.add_argument("--tag", default=None, help="model directory name used in the tree")
    ap.add_argument("--benchmark", default="ruler", choices=("ruler", "jsonl"))
    ap.add_argument("--jsonl", default=None, help="input JSONL when --benchmark jsonl")
    ap.add_argument("--ruler-root", default="data/raw/ruler")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384])
    ap.add_argument("--tasks", nargs="+", default=[
        "niah_single_2", "niah_multikey_1", "niah_multivalue",
        "niah_multiquery", "cwe", "fwe",
    ])
    ap.add_argument("--max-tokens", type=int, default=262144,
                    help="skip prompts longer than this (model native limit)")
    ap.add_argument("--quant", default="q8", choices=("fp16", "q8"))
    ap.add_argument("--block-size", type=int, default=8192,
                    help="K scale block size (must match the decode reader)")
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="tokens per forward chunk; lower this on OOM")
    ap.add_argument("--limit", type=int, default=0,
                    help="max samples per (length, task); 0 = all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="on the first sample, compare q8 logits against fp16")
    ap.add_argument("--sync-write", action="store_true",
                    help="write in the main thread instead of a writer thread")
    args = ap.parse_args()

    if args.benchmark == "jsonl" and not args.jsonl:
        raise SystemExit("--benchmark jsonl requires --jsonl")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text_cfg = model.config.get_text_config(decoder=True)
    layer_types = list(text_cfg.layer_types)
    num_layers = text_cfg.num_hidden_layers

    tag = args.tag or f"{Path(args.model).name}-bf16"
    out_root = Path(args.out)
    executor = None if args.sync_write else ThreadPoolExecutor(max_workers=1)
    pending = []

    def check_pending() -> None:
        for future in list(pending):
            if future.done():
                pending.remove(future)
                exc = future.exception()
                if exc is not None:
                    raise exc

    if args.benchmark == "ruler":
        samples = load_ruler(Path(args.ruler_root), args.lengths, args.tasks, args.limit)
    else:
        samples = load_jsonl(Path(args.jsonl), args.limit)

    built = skipped = 0
    for sample in samples:
        ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
        if not ids:
            print(f"[skip] {sample['sample_id']}: empty prompt")
            skipped += 1
            continue
        if len(ids) > args.max_tokens:
            print(f"[skip] {sample['sample_id']}: {len(ids)} > {args.max_tokens} tokens")
            skipped += 1
            continue

        path = sample_path(out_root, tag, args.quant, args.benchmark,
                           sample["length"], sample["task"], sample["sample_id"])
        if is_complete(path, num_layers) and not args.overwrite:
            skipped += 1
            continue

        print(f"[build] {sample['sample_id']} tokens={len(ids)} -> {path}")
        cache = HybridQuantCache(text_cfg, args.quant, args.block_size)
        dense_prefill(model, torch.tensor([ids], dtype=torch.long, device="cuda"),
                      cache, args.chunk_size)

        meta = {
            "sample_id": sample["sample_id"],
            "benchmark": args.benchmark,
            "task": sample["task"],
            "length": sample["length"],
            "seq_len": len(ids),
            "model": str(args.model),
            "dtype": "bfloat16",
            "quant": args.quant,
            "chunk_size": args.chunk_size,
            "block_size": args.block_size,
            "num_layers": num_layers,
            "layer_types": layer_types,
            "num_kv_heads": text_cfg.num_key_value_heads,
            "head_dim": text_cfg.head_dim,
            "rope_parameters": text_cfg.rope_parameters,
            "linear": {
                "conv_kernel": text_cfg.linear_conv_kernel_dim,
                "num_key_heads": text_cfg.linear_num_key_heads,
                "num_value_heads": text_cfg.linear_num_value_heads,
                "key_head_dim": text_cfg.linear_key_head_dim,
                "value_head_dim": text_cfg.linear_value_head_dim,
            },
            "scheme": {
                "k": ("fp16" if args.quant == "fp16"
                      else f"per-(head, block, channel) int8, block={args.block_size}"),
                "v": "fp16" if args.quant == "fp16" else "per-token int8",
                "linear": "conv_state fp16 + recurrent_state fp32",
            },
        }
        layers = cache.snapshot(layer_types)
        if executor is None:
            save_sample(path, layers, meta, ids)
        else:
            check_pending()
            pending.append(executor.submit(save_sample, path, layers, meta, ids))

        built += 1
        print(f"  peak GPU: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

        if args.verify and built == 1:
            verify(model, text_cfg, ids, args.quant, args.block_size)

        del cache, layers
        torch.cuda.empty_cache()

    if executor is not None:
        executor.shutdown(wait=True)
        check_pending()
    print(f"done: built={built} skipped={skipped}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
