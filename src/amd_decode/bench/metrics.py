"""Official RULER string-match metrics and a small registry.

RULER's ``eval/synthetic/constants.py`` defines:

* ``string_match_all``  — fraction of every reference string found in the
  prediction; used by NIAH, variable tracking, common/frequent word extraction.
* ``string_match_part`` — 1 if any reference string is found; used by QA.

Both are case-insensitive substring checks, exactly as in the reference code.
Scores here are in [0, 1]; :class:`EvalReport` multiplies by 100.
"""

from __future__ import annotations

from typing import Callable, Sequence

Metric = Callable[[str, Sequence[str]], float]


def string_match_all(prediction: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    pred = prediction.lower()
    return sum(1.0 for ref in references if ref.lower() in pred) / len(references)


def string_match_part(prediction: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    pred = prediction.lower()
    return 1.0 if any(ref.lower() in pred for ref in references) else 0.0


METRICS: dict[str, Metric] = {
    "string_match_all": string_match_all,
    "string_match_part": string_match_part,
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown metric {name!r}; known: {sorted(METRICS)}"
        ) from exc
