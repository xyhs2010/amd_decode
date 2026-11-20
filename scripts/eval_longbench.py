"""LongBench v2 accuracy over the saved prefill caches, reported per length.

Run from the repo root:

    python scripts/eval_longbench.py --impl hip
    python scripts/eval_longbench.py --impl device --lengths 32768 131072

Only samples that already have a KV cache are scored (the caches under
``data/kv_cache``), against the labels in ``metadata.jsonl``.  Each length group
prints its summary as soon as it finishes.

The current caches use the raw official template.  For chat-template caches
(``enable_thinking`` at prefill time) the Qwen3.5-9B card recommends these
sampling settings, with ``repetition_penalty=1.0`` throughout:

    thinking, general      temperature 1.0  top_p 0.95  top_k 20  min_p 0  presence 1.5
    thinking, precise code           0.6         0.95        20        0            0.0
    non-thinking, general            0.7         0.8         20        0            1.5
    non-thinking, reasoning          1.0         1.0         40        0            2.0

``max_new_tokens`` 32768 (81920 for hard benchmarks); ``presence_penalty``
0..2 only damps endless repetition.  This script generates greedily (argmax,
no sampling); a sampled run needs ``transformers.generate`` with those kwargs.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from amd_decode.bench import EvalConfig, EvalReport, QwenTokenizer, evaluate
from amd_decode.bench.longbench import load_longbench_samples
from amd_decode.utils import SavedKVCache
from amd_decode.with_hip import HipQwen35
from amd_decode.with_sparse import NaiveIvfPolicy, SparseQwen35
from amd_decode.with_torch import Qwen35Torch

IMPLS = {"device": Qwen35Torch, "hip": HipQwen35, "sparse": SparseQwen35}


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
    parser.add_argument("--index-dir", default="data/ivf",
                        help="sparse only: IVF indices written by "
                             "scripts/build_ivf_index.py")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--nothink", action="store_true",
                        help="chat-template caches (thinking on): feed the 3 tokens "
                             "that close the empty think block before decoding")
    args = parser.parse_args()

    samples = load_longbench_samples(
        args.cache_root, args.metadata, lengths=tuple(args.lengths))
    if args.nothink:
        samples = [replace(s, answer_prefix="\n</think>\n\n") for s in samples]
    excluded = set(args.exclude)
    samples = [s for s in samples if s.sample_id not in excluded]
    if not samples:
        raise SystemExit("no saved LongBench v2 caches found")

    tokenizer = QwenTokenizer(args.model_dir)
    decoder = IMPLS[args.impl](args.model_dir, device="cuda", dtype=torch.float16)
    config = EvalConfig(max_new_tokens=args.max_new_tokens, verbose=True)

    def prepare(sample) -> None:
        saved = SavedKVCache(sample.cache_path)
        if isinstance(decoder, SparseQwen35):
            # One index per sample: loaded if already built, else built here
            # (very slow at long contexts - run build_ivf_index.py first).
            decoder.policy = NaiveIvfPolicy(
                index_file=Path(args.index_dir) / f"{sample.sample_id}.pt")
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
