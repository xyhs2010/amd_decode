"""Accuracy evaluation on prefill-machine KV caches.

This package is about *quality*, not speed:

* :mod:`amd_decode.bench.tokenizer` — dependency-free Qwen3 tokenizer;
* :mod:`amd_decode.bench.ruler` — RULER adapter over ``data/kv_cache``;
* :mod:`amd_decode.bench.generate` — greedy decode on a saved prefix;
* :mod:`amd_decode.bench.runner` — evaluate + score;
* :mod:`amd_decode.bench.types` — samples, results and the report;
* :mod:`amd_decode.bench.metrics` — official RULER string-match metrics.

The runner is benchmark-agnostic. A new benchmark only has to produce
``EvalSample`` objects (saved cache + answer prefix + references + metric);
the decode backend only has to implement
``forward(input_ids, kv=None, position_offset=0)``.
"""

from .generate import DecodeBackend, SavedPrefixKVCache, generate
from .metrics import METRICS, get_metric, string_match_all, string_match_part
from .ruler import discover_ruler_caches, load_ruler_samples, ruler_task_spec
from .runner import EvalConfig, evaluate
from .tokenizer import QwenTokenizer
from .types import EvalReport, EvalSample, Prediction, SampleResult

__all__ = [
    "DecodeBackend",
    "EvalConfig",
    "EvalReport",
    "EvalSample",
    "METRICS",
    "Prediction",
    "QwenTokenizer",
    "SampleResult",
    "SavedPrefixKVCache",
    "discover_ruler_caches",
    "evaluate",
    "generate",
    "get_metric",
    "load_ruler_samples",
    "ruler_task_spec",
    "string_match_all",
    "string_match_part",
]
