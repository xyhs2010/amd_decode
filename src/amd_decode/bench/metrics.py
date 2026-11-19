"""Accuracy metrics and a small registry.

RULER's ``eval/synthetic/constants.py`` defines:

* ``string_match_all``  — fraction of every reference string found in the
  prediction; used by NIAH, variable tracking, common/frequent word extraction.
* ``string_match_part`` — 1 if any reference string is found; used by QA.

Both are case-insensitive substring checks, exactly as in the reference code.
``letter_match`` scores LongBench v2: the prediction's choice letter must equal
the label.  Scores here are in [0, 1]; :class:`EvalReport` multiplies by 100.
"""

from __future__ import annotations

import re
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


_ANSWER = re.compile(r"The correct answer is \(([A-D])\)")
_LETTER = re.compile(r"\(([A-D])\)")
_BARE_LETTER = re.compile(r"\b([A-D])\b")


def letter_match(prediction: str, references: Sequence[str]) -> float:
    """LongBench v2: the predicted choice letter must match the label.

    A reasoning model often names choices on the way ("option (B) says ..."),
    so the *last* answer wins: the final formatted line if there is one,
    otherwise the last parenthesised or bare letter.
    """
    if not references:
        return 0.0
    letters = (
        _ANSWER.findall(prediction)
        or _LETTER.findall(prediction)
        or _BARE_LETTER.findall(prediction.upper())
    )
    return 1.0 if letters and letters[-1].upper() == references[0].upper() else 0.0


METRICS: dict[str, Metric] = {
    "letter_match": letter_match,
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
