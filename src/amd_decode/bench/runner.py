"""Run an accuracy evaluation over ``EvalSample`` objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch

from .. import DecodeModel
from .generate import generate
from .metrics import get_metric
from .types import EvalReport, EvalSample, SampleResult


@dataclass
class EvalConfig:
    """Evaluation knobs. There is deliberately no timing/throughput here."""

    #: Override every sample's ``max_new_tokens`` (debug only; changes scores).
    max_new_tokens: int | None = None
    empty_cache: bool = True
    verbose: bool = True


def _preview(text: str, limit: int = 90) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def evaluate(
    decoder: DecodeModel,
    tokenizer,
    samples: Iterable[EvalSample],
    *,
    config: EvalConfig | None = None,
    prepare: Optional[Callable[[EvalSample], None]] = None,
    on_result: Optional[Callable[[SampleResult], None]] = None,
) -> EvalReport:
    """Greedy-decode and score every sample.

    ``prepare(sample)`` is called before each sample and is where the caller
    attaches that sample's cache (``decoder.set_kv_cache(...)``).  ``on_result``
    is called after each sample, which lets a script append results to disk and
    resume a long run.
    """
    config = config or EvalConfig()
    samples = list(samples)
    if not samples:
        raise ValueError("samples must not be empty")

    benchmarks = {sample.benchmark for sample in samples}
    report = EvalReport(benchmark=benchmarks.pop() if len(benchmarks) == 1 else "mixed")

    for index, sample in enumerate(samples, start=1):
        if prepare is not None:
            prepare(sample)
        try:
            prediction = generate(
                decoder,
                tokenizer,
                sample,
                max_new_tokens=config.max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"{sample.sample_id}: out of GPU memory during prefill/decode. "
                "Try a shorter --lengths, a smaller --limit, or q4 KV caches."
            ) from exc

        score = get_metric(sample.metric)(prediction.text, sample.references)
        result = SampleResult(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            task=sample.task,
            length=sample.length,
            metric=sample.metric,
            prediction=prediction.text,
            references=sample.references,
            score=score,
            generated_tokens=prediction.num_generated,
        )
        report.add(result)
        if on_result is not None:
            on_result(result)
        if config.verbose:
            print(
                f"[{index}/{len(samples)}] {sample.task} L{sample.length} "
                f"{sample.sample_id.rsplit('_', 1)[-1]}: "
                f"score={score:.2f} pred={_preview(prediction.text)!r}"
            )
        if config.empty_cache and getattr(decoder, "device") == "cuda":
            torch.cuda.empty_cache()

    return report
