"""Dependency-free Qwen3 tokenizer for evaluation.

Only the two operations the evaluator needs are implemented:

* :meth:`QwenTokenizer.encode` — text to token ids (answer prefixes, and whole
  prompts when verifying a saved cache);
* :meth:`QwenTokenizer.decode` — generated ids back to text.

It mirrors ``tokenizer.json``: NFC normalization, the Qwen/GPT-4 pre-tokenizer
regex, the ByteLevel byte-to-unicode map and the BPE merge ranks. The
pre-tokenizer's ``\\p{L}`` / ``\\p{N}`` classes are approximated with Python's
Unicode-aware ``re`` (``[^\\W\\d_]`` for letters, ``\\d`` for decimal digits);
this is exact for the English RULER prompts and is validated against every
saved cache as follows::

    tokens = QwenTokenizer(model_dir).encode(row["input"])
    assert tokens == SavedKVCache(sample_dir).tokens().tolist()

No ``transformers``/``tokenizers``/``regex`` dependency is added.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 ByteLevel byte -> printable unicode character table."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, chars)))


# Qwen/GPT-4 pre-tokenizer. ``\p{L}`` -> [^\W\d_], ``\p{N}`` -> \d.
_LETTER = r"[^\W\d_]"  # Unicode letter (word char minus digit/underscore)
_SPLIT_RE = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    rf"|(?:[^\r\n\w]|_)?{_LETTER}+"  # [^\r\n\p{L}\p{N}]?\p{L}+
    r"|\d"  # \p{N}
    r"| ?(?:[^\s\w]|_)+[\r\n]*"  #  ?[^\s\p{L}\p{N}]+[\r\n]*
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class QwenTokenizer:
    """Byte-level BPE tokenizer backed by ``tokenizer.json``."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        data = json.loads((model_dir / "tokenizer.json").read_text())

        self._vocab: dict[str, int] = data["model"]["vocab"]
        self._id_to_token: dict[int, str] = {i: s for s, i in self._vocab.items()}
        self._ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(data["model"]["merges"]):
            first, second = merge.split(" ", 1)
            self._ranks[(first, second)] = rank

        self._added: dict[int, str] = {
            int(tok["id"]): tok["content"] for tok in data.get("added_tokens", [])
        }
        self._added_by_content = {text: idx for idx, text in self._added.items()}
        self._special_re = (
            re.compile(
                "|".join(
                    re.escape(text)
                    for text in sorted(self._added_by_content, key=len, reverse=True)
                )
            )
            if self._added_by_content
            else None
        )

        self._b2u = _bytes_to_unicode()
        self._u2b = {char: byte for byte, char in self._b2u.items()}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}

        generation_config = model_dir / "generation_config.json"
        eos = 151643
        if generation_config.exists():
            eos = json.loads(generation_config.read_text()).get("eos_token_id", eos)
        self.eos_token_ids = {int(v) for v in (eos if isinstance(eos, list) else [eos])}
        self.eos_token_id = min(self.eos_token_ids)

    # ------------------------------------------------------------------ encode

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode ``text``.

        ``add_special_tokens`` is accepted for API compatibility but ignored:
        the Qwen3 Base tokenizer has no BOS/EOS template, and the saved KV
        caches were built with ``add_special_tokens=False``.
        """
        del add_special_tokens
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []

        if self._special_re is None:
            self._encode_text(text, ids)
            return ids

        position = 0
        for match in self._special_re.finditer(text):
            if match.start() > position:
                self._encode_text(text[position : match.start()], ids)
            ids.append(self._added_by_content[match.group(0)])
            position = match.end()
        if position < len(text):
            self._encode_text(text[position:], ids)
        return ids

    def _encode_text(self, text: str, ids: list[int]) -> None:
        position = 0
        for match in _SPLIT_RE.finditer(text):
            if match.start() > position:
                # Defensive: never silently drop text if the regex leaves a gap.
                self._encode_piece(text[position : match.start()], ids)
            self._encode_piece(match.group(0), ids)
            position = match.end()
        if position < len(text):
            self._encode_piece(text[position:], ids)

    def _encode_piece(self, piece: str, ids: list[int]) -> None:
        if not piece:
            return
        # ByteLevel: UTF-8 bytes, then the byte -> unicode table.
        byte_text = "".join(self._b2u[byte] for byte in piece.encode("utf-8"))
        for token in self._bpe(byte_text):
            ids.append(self._vocab[token])

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token)
        while len(word) > 1:
            best_pair: tuple[str, str] | None = None
            best_rank: int | None = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            first, second = best_pair
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        self._bpe_cache[token] = word
        return word

    # ------------------------------------------------------------------ decode

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        """Decode ids, mapping token strings back through the ByteLevel table."""
        parts: list[str] = []
        pending = bytearray()

        def flush() -> None:
            if pending:
                parts.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for token_id in ids:
            token_id = int(token_id)
            if token_id in self._added:
                if not skip_special_tokens:
                    flush()
                    parts.append(self._added[token_id])
                continue
            token = self._id_to_token.get(token_id)
            if token is None:
                continue
            for char in token:
                pending.append(self._u2b[char])
        flush()
        return "".join(parts)
