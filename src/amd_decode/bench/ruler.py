"""RULER adapter: saved KV cache sample + raw jsonl label -> ``EvalSample``.

The saved caches contain the prompt (``input``) only. RULER's official
generation prompt is ``input + answer_prefix``, so the evaluator feeds the
``answer_prefix`` at decode time; it is *not* part of the prefill.

Reference: ``data/raw/ruler_repo/scripts/pred/call_api.py`` and
``scripts/data/synthetic/constants.py``. ``max_new_tokens`` and the metric per
task are taken from the official constants.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..utils.prefill_cache import SavedKVCache
from .types import EvalSample


@dataclass(frozen=True)
class RulerTaskSpec:
    max_new_tokens: int
    metric: str


# Official synthetic/constants.py and hydra_llm's loaders.py.
_TASK_SPECS = {
    "niah": RulerTaskSpec(128, "string_match_all"),
    "variable_tracking": RulerTaskSpec(30, "string_match_all"),
    "common_words_extraction": RulerTaskSpec(120, "string_match_all"),
    "freq_words_extraction": RulerTaskSpec(50, "string_match_all"),
    "qa": RulerTaskSpec(32, "string_match_part"),
}


def ruler_task_spec(task: str) -> RulerTaskSpec:
    if task.startswith("niah_"):
        return _TASK_SPECS["niah"]
    if task == "cwe":
        return _TASK_SPECS["common_words_extraction"]
    if task == "fwe":
        return _TASK_SPECS["freq_words_extraction"]
    if task == "vt":
        return _TASK_SPECS["variable_tracking"]
    if task.startswith("qa_"):
        return _TASK_SPECS["qa"]
    raise ValueError(f"unknown RULER task {task!r}")


def _sample_index(sample_id: str, meta: dict) -> int:
    if "index" in meta:
        return int(meta["index"])
    try:
        return int(sample_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(
            f"cannot derive RULER index from sample_id {sample_id!r}"
        ) from exc


def _raw_rows(ruler_root: Path, length: int, task: str) -> dict[int, list[dict]]:
    path = ruler_root / f"L{length}" / task / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing RULER label file: {path}")
    rows: dict[int, list[dict]] = {}
    for line in path.open():
        row = json.loads(line)
        rows.setdefault(int(row["index"]), []).append(row)
    return rows


def _cached_tokens(path: Path) -> list[int]:
    return SavedKVCache(path).tokens().tolist()


def _select_row(
    path: Path,
    rows: list[dict],
    tokenizer,
    *,
    verify: bool,
) -> dict:
    """Pick the jsonl row that was actually prefilled.

    RULER's jsonl can contain duplicate ``index`` values. The cache builder
    skips a path that is already complete, so the first duplicate is the one
    cached. When a tokenizer is available, matching the encoded prompt against
    ``tokens.safetensors`` disambiguates and also verifies the tokenizer.
    """
    if len(rows) == 1 and not verify:
        return rows[0]
    if tokenizer is None:
        if len(rows) > 1:
            warnings.warn(
                f"{path.name}: duplicate index; using the first row "
                "(matches the cache builder's skip-if-complete order)",
                stacklevel=2,
            )
        return rows[0]

    cached = _cached_tokens(path)
    matches = []
    for row in rows:
        encoded = tokenizer.encode(row["input"], add_special_tokens=False)
        if encoded == cached:
            matches.append(row)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"{path}: no jsonl row tokenizes to the cached prompt "
            f"({len(rows)} candidate(s)); the tokenizer or label file is wrong"
        )
    warnings.warn(
        f"{path.name}: {len(matches)} rows tokenize identically; using the first",
        stacklevel=2,
    )
    return matches[0]


def discover_ruler_caches(kv_root: str | Path) -> list[Path]:
    """Return sample directories containing ``meta.json`` with benchmark=ruler."""
    root = Path(kv_root)
    metas = [root / "meta.json"] if (root / "meta.json").exists() else list(
        root.glob("**/meta.json")
    )
    if not metas:
        raise FileNotFoundError(f"no meta.json found under {root}")

    paths: list[Path] = []
    for meta_path in sorted(metas):
        meta = json.loads(meta_path.read_text())
        if meta.get("benchmark") == "ruler":
            paths.append(meta_path.parent)
    if not paths:
        raise FileNotFoundError(f"no RULER KV caches found under {root}")
    return paths


def load_ruler_samples(
    kv_root: str | Path,
    ruler_root: str | Path = "data/raw/ruler",
    *,
    tokenizer=None,
    lengths: Iterable[int] | None = None,
    tasks: Iterable[str] | None = None,
    limit: int | None = None,
    per_task_limit: int | None = None,
    verify_tokens: bool = False,
) -> list[EvalSample]:
    """Build ``EvalSample`` objects from saved caches and raw RULER labels.

    Args:
        kv_root: tree containing ``<model>/<quant>/ruler/L*/<task>/<sample>/``.
        ruler_root: tree containing ``L*/<task>/test.jsonl``.
        tokenizer: optional :class:`QwenTokenizer`; enables token-exact
            disambiguation of duplicate indices and ``verify_tokens``.
        lengths / tasks: filters on ``meta["length"]`` / ``meta["task"]``.
        limit: keep at most N samples overall.
        per_task_limit: keep at most N samples per ``(length, task)`` pair.
        verify_tokens: re-encode each prompt and require it to match
            ``tokens.safetensors``. Costs ~0.1 s per 16K prompt.

    Returns samples sorted by ``(length, task, index)``.
    """
    if verify_tokens and tokenizer is None:
        raise ValueError("verify_tokens=True requires a tokenizer")

    length_set = {int(v) for v in lengths} if lengths is not None else None
    task_set = set(tasks) if tasks is not None else None

    ruler_root = Path(ruler_root)
    raw_cache: dict[tuple[int, str], dict[int, list[dict]]] = {}
    samples: list[EvalSample] = []

    for path in discover_ruler_caches(kv_root):
        meta = json.loads((path / "meta.json").read_text())
        if not meta.get("benchmark") == "ruler":
            continue
        task = str(meta.get("task", ""))
        length = int(meta.get("length", 0))
        if length_set is not None and length not in length_set:
            continue
        if task_set is not None and task not in task_set:
            continue

        sample_id = str(meta.get("sample_id", path.name))
        index = _sample_index(sample_id, meta)
        key = (length, task)
        if key not in raw_cache:
            raw_cache[key] = _raw_rows(ruler_root, length, task)
        rows = raw_cache[key].get(index)
        if not rows:
            raise ValueError(
                f"{sample_id}: index {index} not found in "
                f"{ruler_root / f'L{length}' / task / 'test.jsonl'}"
            )

        row = _select_row(path, rows, tokenizer, verify=verify_tokens)
        answer_prefix = row.get("answer_prefix")
        if not answer_prefix:
            raise ValueError(f"{sample_id}: raw row has no answer_prefix")
        references = tuple(str(value) for value in row["outputs"])
        spec = ruler_task_spec(task)

        samples.append(
            EvalSample(
                sample_id=sample_id,
                benchmark="ruler",
                task=task,
                length=length,
                cache_path=path,
                answer_prefix=answer_prefix,
                references=references,
                max_new_tokens=spec.max_new_tokens,
                metric=spec.metric,
                metadata={
                    "index": index,
                    "seq_len": int(meta.get("seq_len", 0)),
                    "needle_positions": row.get("needle_positions"),
                    "needle_depths": row.get("needle_depths"),
                },
            )
        )

    samples.sort(key=lambda s: (s.length, s.task, int(s.metadata["index"])))

    if per_task_limit is not None:
        if per_task_limit < 1:
            raise ValueError("per_task_limit must be >= 1")
        kept: list[EvalSample] = []
        seen: dict[tuple[int, str], int] = {}
        for sample in samples:
            key = (sample.length, sample.task)
            count = seen.get(key, 0)
            if count < per_task_limit:
                kept.append(sample)
                seen[key] = count + 1
        samples = kept

    if limit is not None:
        samples = samples[:limit]

    if not samples:
        raise ValueError("no RULER samples matched the given filters")
    return samples
