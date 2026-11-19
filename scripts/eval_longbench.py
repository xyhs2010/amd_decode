"""LongBench v2 accuracy over the saved prefill caches, reported per length.

Run from the repo root:

    python scripts/eval_longbench.py --impl hip
    python scripts/eval_longbench.py --impl device --lengths 32768 131072

Only samples that already have a KV cache are scored (the caches under
``data/kv_cache``), against the labels in ``metadata.jsonl``.  Each length group
prints its summary as soon as it finishes.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

from amd_decode.bench import EvalConfig, EvalReport, QwenTokenizer, evaluate
from amd_decode.bench.longbench import load_longbench_samples
from amd_decode.utils import SavedKVCache
from amd_decode.with_hip import HipQwen35
from amd_decode.with_torch import Qwen35Torch

IMPLS = {"device": Qwen35Torch, "hip": HipQwen35}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", default="hip", choices=sorted(IMPLS))
    parser.add_argument("--model-dir", default="data/models/Qwen3.5-9B")
    parser.add_argument("--cache-root",
                        default="data/kv_cache/Qwen3.5-9B/bfloat16/jsonl")
    parser.add_argument("--metadata", default="data/raw/longbench_v2/metadata.jsonl")
    parser.add_argument("--lengths", type=int, nargs="+", default=[])
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="sample ids to skip (e.g. ones already scored)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    samples = load_longbench_samples(
        args.cache_root, args.metadata, lengths=tuple(args.lengths))
    excluded = set(args.exclude)
    samples = [s for s in samples if s.sample_id not in excluded]
    if not samples:
        raise SystemExit("no saved LongBench v2 caches found")

    tokenizer = QwenTokenizer(args.model_dir)
    decoder = IMPLS[args.impl](args.model_dir, device="cuda", dtype=torch.float16)
    config = EvalConfig(max_new_tokens=args.max_new_tokens, verbose=True)

    def prepare(sample) -> None:
        saved = SavedKVCache(sample.cache_path)
        prefix = len(tokenizer.encode(sample.answer_prefix, add_special_tokens=False))
        decoder.load_prefill(
            saved, saved.seq_len + prefix + sample.max_new_tokens + 2)

    by_length: dict[int, list] = defaultdict(list)
    for sample in samples:
        by_length[sample.length].append(sample)
    print(f"{len(samples)} cached samples, lengths {sorted(by_length)}", flush=True)

    overall = EvalReport(benchmark="longbench_v2")
    for length in sorted(by_length):
        group = by_length[length]
        print(f"\n== L{length}: {len(group)} samples ==", flush=True)
        report = EvalReport(benchmark="longbench_v2")
        skipped: list[str] = []
        for sample in group:
            try:
                part = evaluate(decoder, tokenizer, [sample], config=config,
                                prepare=prepare)
            except Exception as exc:
                print(f"SKIP {sample.sample_id}: {type(exc).__name__}: {exc}",
                      flush=True)
                skipped.append(sample.sample_id)
                decoder.load_prefill(None)
                torch.cuda.empty_cache()
                continue
            report.results.extend(part.results)
        overall.results.extend(report.results)
        print(report.summary(), flush=True)
        if skipped:
            print(f"skipped {len(skipped)}: {' '.join(skipped)}", flush=True)

    if len(by_length) > 1:
        print("\n== all lengths ==")
        print(overall.summary(), flush=True)


if __name__ == "__main__":
    main()
