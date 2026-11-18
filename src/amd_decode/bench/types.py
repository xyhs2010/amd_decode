"""Data types shared by the accuracy evaluator.

The evaluator is benchmark-agnostic: a benchmark adapter produces
:class:`EvalSample` objects, the runner turns them into :class:`SampleResult`
objects, and :class:`EvalReport` aggregates scores. Today the only adapter is
RULER (see :mod:`amd_decode.bench.ruler`); adding another benchmark means
producing the same ``EvalSample`` stream and registering a metric.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvalSample:
    """One evaluation sample whose prompt KV is already on disk.

    ``cache_path`` points to a ``SavedKVCache`` directory (``meta.json``,
    ``tokens.safetensors``, ``layer_NN.safetensors``). ``answer_prefix`` is
    appended after the cached prompt and is not part of the prefill, matching
    the official RULER protocol (``input + answer_prefix``).
    """

    sample_id: str
    benchmark: str
    task: str
    length: int
    cache_path: Path
    answer_prefix: str
    references: tuple[str, ...]
    max_new_tokens: int
    metric: str
    stop: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    text: str
    generated_ids: tuple[int, ...]

    @property
    def num_generated(self) -> int:
        return len(self.generated_ids)


@dataclass(frozen=True)
class SampleResult:
    """Score for one sample. ``score`` is in [0, 1]."""

    sample_id: str
    benchmark: str
    task: str
    length: int
    metric: str
    prediction: str
    references: tuple[str, ...]
    score: float
    generated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "task": self.task,
            "length": self.length,
            "metric": self.metric,
            "prediction": self.prediction,
            "references": list(self.references),
            "score": self.score,
            "generated_tokens": self.generated_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SampleResult":
        return cls(
            sample_id=str(data["sample_id"]),
            benchmark=str(data.get("benchmark", "")),
            task=str(data["task"]),
            length=int(data["length"]),
            metric=str(data.get("metric", "")),
            prediction=str(data.get("prediction", "")),
            references=tuple(str(v) for v in data.get("references", ())),
            score=float(data.get("score", 0.0)),
            generated_tokens=int(data.get("generated_tokens", 0)),
        )


def _mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) * 100.0 if values else float("nan")


@dataclass
class EvalReport:
    """Per-sample results plus the aggregate accuracy report."""

    results: list[SampleResult] = field(default_factory=list)
    benchmark: str = ""

    def add(self, result: SampleResult) -> None:
        self.results.append(result)

    @property
    def num_samples(self) -> int:
        return len(self.results)

    def task_scores(self) -> dict[str, float]:
        """Accuracy (%) per task, averaging over that task's samples."""
        groups: dict[str, list[float]] = {}
        for result in self.results:
            groups.setdefault(result.task, []).append(result.score)
        return {task: _mean(scores) for task, scores in sorted(groups.items())}

    def length_scores(self) -> dict[int, float]:
        """Accuracy (%) per context length, averaging over all tasks."""
        groups: dict[int, list[float]] = {}
        for result in self.results:
            groups.setdefault(result.length, []).append(result.score)
        return {length: _mean(scores) for length, scores in sorted(groups.items())}

    def task_length_scores(self) -> dict[tuple[str, int], float]:
        """Accuracy (%) per (task, length) cell."""
        groups: dict[tuple[str, int], list[float]] = {}
        for result in self.results:
            groups.setdefault((result.task, result.length), []).append(result.score)
        return {key: _mean(scores) for key, scores in sorted(groups.items())}

    def overall_score(self) -> float:
        """Mean of the (task, length) cell scores, already in percent.

        With a single context length this is RULER's usual average over tasks;
        with several lengths every cell is weighted equally.
        """
        cells = list(self.task_length_scores().values())
        return statistics.fmean(cells) if cells else float("nan")

    def summary(self) -> str:
        if not self.results:
            return f"{self.benchmark or 'eval'}: no results"

        tasks = sorted({r.task for r in self.results})
        lengths = sorted({r.length for r in self.results})
        cells = self.task_length_scores()

        lines = [
            f"{self.benchmark or 'eval'} accuracy: {self.num_samples} samples, "
            f"{len(tasks)} task(s), {len(lengths)} length(s)",
            "",
        ]

        if len(lengths) == 1:
            length = lengths[0]
            lines.append(f"{'task':<24} {'n':>5} {'score':>8}")
            lines.append("-" * 40)
            for task in tasks:
                count = sum(1 for r in self.results if r.task == task)
                lines.append(
                    f"{task:<24} {count:>5} {cells.get((task, length), float('nan')):>8.2f}"
                )
            lines.append("-" * 40)
            lines.append(
                f"{'overall':<24} {self.num_samples:>5} {self.overall_score():>8.2f}"
            )
        else:
            header = f"{'task':<24} {'n':>5}" + "".join(
                f" {('L' + str(length)):>10}" for length in lengths
            )
            lines.append(header)
            lines.append("-" * len(header))
            for task in tasks:
                count = sum(1 for r in self.results if r.task == task)
                row = f"{task:<24} {count:>5}"
                for length in lengths:
                    score = cells.get((task, length))
                    row += f" {score:>10.2f}" if score is not None else f" {'-':>10}"
                lines.append(row)
            lines.append("-" * len(header))
            row = f"{'overall':<24} {self.num_samples:>5}"
            for length in lengths:
                score = self.length_scores().get(length)
                row += f" {score:>10.2f}" if score is not None else f" {'-':>10}"
            lines.append(row)

        lines.append("")
        lines.append(f"overall: {self.overall_score():.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "num_samples": self.num_samples,
            "overall": self.overall_score(),
            "task_scores": self.task_scores(),
            "length_scores": {str(k): v for k, v in self.length_scores().items()},
            "task_length_scores": [
                {
                    "task": task,
                    "length": length,
                    "score": score,
                    "num_samples": sum(
                        1
                        for r in self.results
                        if r.task == task and r.length == length
                    ),
                }
                for (task, length), score in self.task_length_scores().items()
            ],
            "results": [result.to_dict() for result in self.results],
        }
