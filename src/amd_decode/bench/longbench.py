"""LongBench v2 adapter: labeled metadata + saved caches -> ``EvalSample``.

The cached prompt is the official template (``prefill/build_longbench_v2.py``),
which asks for ``The correct answer is (X)``.  The model reasons first, so it
gets a large budget and :func:`amd_decode.bench.metrics.letter_match` scores the
last formatted answer against ``metadata.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import EvalSample

BENCHMARK = "longbench_v2"
MAX_NEW_TOKENS = 512
ANSWER_PREFIX = "\n"


def load_metadata(path: str | Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in map(json.loads, handle)}


def load_longbench_samples(
    cache_root: str | Path,
    metadata_path: str | Path,
    *,
    lengths: tuple[int, ...] = (),
) -> list[EvalSample]:
    """Every saved cache under ``cache_root/L*/<task>/<sample_id>/`` with a label."""
    labels = load_metadata(metadata_path)
    samples = []
    for path in Path(cache_root).glob("L*/*/*/meta.json"):
        sample_id = path.parent.name
        length = int(path.parent.parent.parent.name[1:])
        if lengths and length not in lengths:
            continue
        row = labels[sample_id]
        samples.append(
            EvalSample(
                sample_id=sample_id,
                benchmark=BENCHMARK,
                task=row["task"],
                length=length,
                cache_path=path.parent,
                answer_prefix=ANSWER_PREFIX,
                references=(row["answer"],),
                max_new_tokens=MAX_NEW_TOKENS,
                metric="letter_match",
            )
        )
    return sorted(samples, key=lambda s: (s.length, s.task, s.sample_id))
