#!/usr/bin/env python
"""Accuracy (not speed) evaluation of Qwen3Torch on RULER saved KV caches.

Run from the repo root (the package is not installed into the venv)::

    # quick default: L16384, 5 samples per (task, length), greedy, full token budget
    PYTHONPATH=src .venv/bin/python scripts/eval_ruler.py

    # one sample per task at every saved length, resumable
    PYTHONPATH=src .venv/bin/python scripts/eval_ruler.py \
        --lengths all --per-task-limit 1 --resume

    # list the samples that would run, without loading model weights
    PYTHONPATH=src .venv/bin/python scripts/eval_ruler.py --list --lengths all

The protocol follows the official RULER code: after the cached prompt, feed
``answer_prefix`` and greedily decode ``tokens_to_generate`` tokens (128 for
NIAH, 120 for cwe, 50 for fwe). Scores use RULER's case-insensitive substring
match. Each prediction is appended to ``--output`` as one JSON line, so a long
run can be resumed with ``--resume``.

Only RULER has saved KV caches today. Other benchmarks need their prompt KV
prefilled on the prefill machine first (see ``hydra_llm/scripts/build_kv_cache.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from amd_decode.bench import (
    EvalConfig,
    EvalReport,
    QwenTokenizer,
    SampleResult,
    evaluate,
    load_ruler_samples,
)
from amd_decode.with_torch import Qwen3Torch


def _parse_lengths(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    try:
        return [int(part) for part in value.replace(",", " ").split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers or 'all', got {value!r}"
        ) from exc


def _parse_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part for part in value.replace(",", " ").split() if part]


def _parse_dtype(value: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[value]


def _load_existing(path: Path) -> dict[str, SampleResult]:
    results: dict[str, SampleResult] = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                result = SampleResult.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"[warn] skipping malformed line {line_number} in {path}: {exc}")
                continue
            results[result.sample_id] = result
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", default="data/models/Qwen3-8B-Base")
    parser.add_argument("--kv-root", default="data/kv_cache")
    parser.add_argument("--ruler-root", default="data/raw/ruler")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--weight-bits",
        type=int,
        default=4,
        choices=(4, 16),
        help="Qwen3Torch weight quantization (the caches were built with w4)",
    )
    parser.add_argument(
        "--lengths",
        type=_parse_lengths,
        default=[16384],
        metavar="N[,N...]|all",
        help="context lengths to evaluate (default: 16384)",
    )
    parser.add_argument(
        "--tasks",
        type=_parse_list,
        default=None,
        metavar="T[,T...]",
        help="RULER tasks (default: all saved tasks)",
    )
    parser.add_argument("--limit", type=int, default=None, help="max samples overall")
    parser.add_argument(
        "--per-task-limit",
        type=int,
        default=5,
        help="max samples per (length, task) pair (default: 5; 0 disables)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override each task's official generation budget (debug only)",
    )
    parser.add_argument(
        "--output",
        default="data/eval/ruler.jsonl",
        help="one JSON result per line (default: data/eval/ruler.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip samples already present in --output and append new results",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        metavar="PATH",
        help="write the aggregate report (including per-sample results) as JSON",
    )
    parser.add_argument(
        "--verify-tokens",
        action="store_true",
        help="re-encode every prompt and require an exact match with the saved "
        "tokens (also disambiguates duplicate jsonl indices)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list matching samples and exit without loading model weights",
    )
    parser.add_argument("--quiet", action="store_true", help="hide per-sample output")
    args = parser.parse_args(argv)

    per_task_limit = args.per_task_limit if args.per_task_limit else None
    tokenizer = QwenTokenizer(args.model_dir)

    try:
        samples = load_ruler_samples(
            args.kv_root,
            args.ruler_root,
            tokenizer=tokenizer,
            lengths=args.lengths,
            tasks=args.tasks,
            limit=args.limit,
            per_task_limit=per_task_limit,
            verify_tokens=args.verify_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"eval_ruler: {exc}") from exc

    if args.list:
        for sample in samples:
            print(
                f"{sample.sample_id}  task={sample.task} length={sample.length} "
                f"max_new={sample.max_new_tokens} metric={sample.metric} "
                f"refs={len(sample.references)}"
            )
        print(f"{len(samples)} sample(s)")
        return

    output_path = Path(args.output)
    existing = _load_existing(output_path) if args.resume else {}
    selected_ids = {sample.sample_id for sample in samples}
    remaining = [sample for sample in samples if sample.sample_id not in existing]

    print(
        f"eval_ruler: {len(samples)} sample(s), {len(remaining)} to run "
        f"({len(existing)} cached result(s) loaded), "
        f"device={args.device}, dtype={args.dtype}, "
        f"weight_bits={args.weight_bits}"
    )
    if args.max_new_tokens is not None:
        print(
            f"[warn] --max-new-tokens {args.max_new_tokens} overrides the official "
            "per-task budget; scores are not comparable to full-budget RULER"
        )

    new_results: list[SampleResult] = []
    if remaining:
        decoder = Qwen3Torch(
            args.model_dir,
            device=args.device,
            dtype=_parse_dtype(args.dtype),
            weight_bits=args.weight_bits,
        )
        config = EvalConfig(
            device=args.device,
            dtype=_parse_dtype(args.dtype),
            max_new_tokens=args.max_new_tokens,
            verbose=not args.quiet,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume and output_path.exists() else "w"
        with output_path.open(mode, encoding="utf-8") as handle:

            def on_result(result: SampleResult) -> None:
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()

            report = evaluate(
                decoder,
                tokenizer,
                remaining,
                config=config,
                on_result=on_result,
            )
        new_results = report.results

    merged = dict(existing)
    merged.update({result.sample_id: result for result in new_results})
    results = [merged[sample.sample_id] for sample in samples if sample.sample_id in merged]
    report = EvalReport(results=results, benchmark="ruler")

    print()
    print(report.summary())
    print(f"\nresults: {output_path}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
