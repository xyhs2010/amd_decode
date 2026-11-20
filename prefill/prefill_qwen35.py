"""Dense prefill for Qwen3.5-9B, saved with transformers' own cache.

This is a thin script around the stock implementation: one forward (optionally
in chunks) with ``DynamicCache``, then write the cache tensors to disk.  There
is no custom attention, no quantization and no verification here; the saved
cache is exactly what transformers produced, and it is checked on the decode
side.

Output per sample::

    <out>/<model_tag>/<dtype>/<benchmark>/L<length>/<task>/<sample_id>/
        meta.json
        tokens.safetensors              # int32 prompt tokens
        layer_00.safetensors            # linear layer:
            conv_state      [1, 8192, 4]       model dtype
            recurrent_state [1, 32, 128, 128]  fp32
        layer_03.safetensors            # full-attention layer:
            k [1, 4, T, 256]                   model dtype
            v [1, 4, T, 256]                   model dtype

The answer prefix is not prefilled, matching the RULER protocol.

Examples::

    python prefill/prefill_qwen35.py --model data/models/Qwen3.5-9B \
        --benchmark ruler --lengths 16384 --tasks niah_single_2 --limit 1

    python prefill/prefill_qwen35.py --model data/models/Qwen3.5-9B \
        --benchmark jsonl --jsonl data/raw/longbench_v2/prompts_256k.jsonl

    # same, but wrap each prompt in the chat template (thinking on)
    python prefill/prefill_qwen35.py --model data/models/Qwen3.5-9B \
        --benchmark jsonl --jsonl ... --chat --tag Qwen3.5-9B-chat
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "this script needs transformers with Qwen3.5 support"
    ) from exc


# --------------------------------------------------------------------------- samples


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
            "prompt": str(row["prompt"]),
        }
        if limit:
            limit -= 1
            if limit == 0:
                break


# --------------------------------------------------------------------------- prefill / io


@torch.inference_mode()
def dense_prefill(model, ids: torch.Tensor, cache, chunk: int) -> None:
    # logits_to_keep=1 skips the full [T, vocab] lm_head output; the cache is
    # the only thing this script needs.
    if chunk <= 0:
        model(input_ids=ids, past_key_values=cache, use_cache=True,
              logits_to_keep=1)
        return
    for start in range(0, ids.shape[1], chunk):
        model(input_ids=ids[:, start : start + chunk],
              past_key_values=cache, use_cache=True, logits_to_keep=1)


def sample_path(root: Path, tag: str, dtype: str, benchmark: str,
                length: int, task: str, sample_id: str) -> Path:
    return root / tag / dtype / benchmark / f"L{length}" / task / sample_id


def is_complete(path: Path, num_layers: int) -> bool:
    return (path / "meta.json").exists() and all(
        (path / f"layer_{i:02d}.safetensors").exists() for i in range(num_layers)
    )


def save_sample(path: Path, cache, layer_types, meta: dict, tokens: list[int]) -> None:
    """Write the stock DynamicCache layer by layer."""
    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if path.exists():
        shutil.rmtree(path)

    save_file({"tokens": torch.tensor(tokens, dtype=torch.int32)},
              str(tmp / "tokens.safetensors"))
    for layer_idx, layer_type in enumerate(layer_types):
        layer = cache.layers[layer_idx]
        if layer_type == "full_attention":
            tensors = {"k": layer.keys.detach().cpu(),
                       "v": layer.values.detach().cpu()}
        else:
            # transformers 5.13 stores the per-batch state without the batch
            # dim; the reader (and older caches) expect [1, ...].
            conv = layer.conv_states[0].detach().cpu()
            rec = layer.recurrent_states[0].detach().cpu()
            tensors = {
                "conv_state": conv.unsqueeze(0) if conv.dim() == 2 else conv,
                "recurrent_state": rec.unsqueeze(0) if rec.dim() == 3 else rec,
            }
        save_file({name: t.contiguous() for name, t in tensors.items()},
                  str(tmp / f"layer_{layer_idx:02d}.safetensors"))
        del tensors

    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(path)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="data/models/Qwen3.5-9B")
    ap.add_argument("--out", default="data/kv_cache")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--benchmark", default="ruler", choices=("ruler", "jsonl"))
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--ruler-root", default="data/raw/ruler")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384])
    ap.add_argument("--tasks", nargs="+", default=[
        "niah_single_2", "niah_multikey_1", "niah_multivalue",
        "niah_multiquery", "cwe", "fwe",
    ])
    ap.add_argument("--max-tokens", type=int, default=262144,
                    help="skip prompts longer than this")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="tokens per forward; 0 = the whole prompt")
    ap.add_argument("--limit", type=int, default=0,
                    help="max samples per (length, task); 0 = all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda",
                    help="any torch device string for this machine")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16"))
    ap.add_argument("--chat", action="store_true",
                    help="wrap the prompt in the chat template with thinking on")
    args = ap.parse_args()

    if args.benchmark == "jsonl" and not args.jsonl:
        raise SystemExit("--benchmark jsonl requires --jsonl")

    dtype = getattr(torch, args.dtype)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa",
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text_cfg = model.config.get_text_config(decoder=True)
    layer_types = list(text_cfg.layer_types)
    num_layers = text_cfg.num_hidden_layers

    tag = args.tag or Path(args.model).name
    out_root = Path(args.out)

    if args.benchmark == "ruler":
        samples = load_ruler(Path(args.ruler_root), args.lengths, args.tasks, args.limit)
    else:
        samples = load_jsonl(Path(args.jsonl), args.limit)

    built = skipped = 0
    for sample in samples:
        text = sample["prompt"]
        if args.chat:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            print(f"[skip] {sample['sample_id']}: empty prompt")
            skipped += 1
            continue
        if len(ids) > args.max_tokens:
            print(f"[skip] {sample['sample_id']}: {len(ids)} > {args.max_tokens} tokens")
            skipped += 1
            continue

        path = sample_path(out_root, tag, args.dtype, args.benchmark,
                           sample["length"], sample["task"], sample["sample_id"])
        if is_complete(path, num_layers) and not args.overwrite:
            skipped += 1
            continue

        print(f"[build] {sample['sample_id']} tokens={len(ids)} -> {path}")
        cache = DynamicCache(config=text_cfg)
        dense_prefill(model, torch.tensor([ids], dtype=torch.long, device=args.device),
                      cache, args.chunk_size)

        meta = {
            "sample_id": sample["sample_id"],
            "benchmark": args.benchmark,
            "task": sample["task"],
            "length": sample["length"],
            "seq_len": len(ids),
            "model": str(args.model),
            "dtype": args.dtype,
            "chunk_size": args.chunk_size,
            "chat_template": "chat" if args.chat else "raw",
            "enable_thinking": bool(args.chat),
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
        }
        save_sample(path, cache, layer_types, meta, ids)
        built += 1
        del cache
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"done: built={built} skipped={skipped}")


if __name__ == "__main__":
    main()
