"""lmsys-chat-1m adapter.

This adapter intentionally does NOT download anything. The user provides
a local path to the dataset (JSONL or parquet). We expose:

  - `LmsysConfig`: path + sampling knobs
  - `load_sessions(cfg)`: yields canonical `SessionTurn` records
  - `build_trace(cfg, params)`: converts sessions into a WorkloadTrace

Schema assumed (documented here so tests can enforce it):
  {
    "conversation_id": str,
    "model": str,
    "turn": int,
    "language": str,
    "conversation": [
      {"role": "user"|"assistant", "content": str},
      ...
    ],
    "openai_moderation": ...,
    "redacted": bool
  }

A `StubLoader` is provided for tests: it returns a small synthetic set of
session turns matching the schema so parsing/sampling logic is
exercised without the real dataset.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..core import Request
from .trace import InMemoryTrace


@dataclass(frozen=True)
class LmsysConfig:
    local_path: str  # required — no default; user must provide
    max_conversations: int | None = None
    language_filter: tuple[str, ...] = ("en",)
    min_turns: int = 1
    max_turns: int = 16
    seed: int = 0


@dataclass(frozen=True)
class SessionTurn:
    conversation_id: str
    turn_idx: int
    role: str
    content: str
    language: str


@dataclass(frozen=True)
class TraceParams:
    arrival_rate_qps: float
    tokens_per_char: float = 0.25  # only used when tokenizer == "mock"
    max_output_tokens: int = 256
    seed: int = 0
    # "mock" or "tiktoken:<encoding>". Missing tiktoken raises rather than
    # silently falling back, so the mock bias can't be misattributed.
    tokenizer: str = "mock"


class StubLoader:
    """Deterministic synthetic loader for tests.

    Emits `n_convs` conversations, each with `turns_per_conv` turns,
    sharing a common system-prompt prefix so KV reuse is non-trivial.
    """

    def __init__(self, n_convs: int = 5, turns_per_conv: int = 3, seed: int = 0) -> None:
        self.n_convs = n_convs
        self.turns_per_conv = turns_per_conv
        self.seed = seed

    def iter_turns(self) -> Iterator[SessionTurn]:
        rng = random.Random(self.seed)
        system_prompt = "You are a helpful assistant. " * 8
        for c in range(self.n_convs):
            cid = f"conv_{c:04d}"
            for t in range(self.turns_per_conv):
                yield SessionTurn(
                    conversation_id=cid,
                    turn_idx=t,
                    role="user" if t % 2 == 0 else "assistant",
                    content=system_prompt + f" turn {t} payload {rng.random():.6f}",
                    language="en",
                )


def load_sessions(cfg: LmsysConfig) -> Iterator[SessionTurn]:
    """Stream SessionTurn records from a local lmsys-chat-1m dump.

    Supports JSONL (one conversation per line). Parquet support is a
    follow-up — a RuntimeError is raised with a helpful message so users
    don't silently get incorrect data.
    """
    path = Path(cfg.local_path)
    if not path.exists():
        raise FileNotFoundError(
            f"lmsys dataset not found at {path}. Download and unpack it "
            f"manually (see docs) and pass the local path via config."
        )
    if path.suffix == ".parquet":
        raise RuntimeError(
            "Parquet loader not implemented. Convert to JSONL or add a "
            "pyarrow dependency and extend this adapter."
        )
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            language = rec.get("language", "en")
            if cfg.language_filter and language not in cfg.language_filter:
                continue
            convo = rec.get("conversation", [])
            if not (cfg.min_turns <= len(convo) <= cfg.max_turns):
                continue
            cid = str(rec.get("conversation_id", rec.get("id", "")))
            for t, turn in enumerate(convo):
                if turn.get("role") == "user":
                    yield SessionTurn(
                        conversation_id=cid,
                        turn_idx=t,
                        role="user",
                        content=turn.get("content", ""),
                        language=language,
                    )


def _mock_tokens(content: str, tokens_per_char: float) -> tuple[int, ...]:
    """Block-hash mock: stable, no deps, preserves prefix structure.

    Yields token ids where shared *content* prefixes produce shared *token*
    prefixes. Under-estimates real token count (English cl100k_base is
    ~0.75 tokens/char; default here is 0.25).
    """
    n = max(1, int(len(content) * tokens_per_char))
    tokens: list[int] = []
    block = 16
    for start in range(0, n, block):
        chunk = content[start : start + 64]
        h = int.from_bytes(hashlib.blake2b(chunk.encode(), digest_size=4).digest(), "big")
        for k in range(min(block, n - start)):
            tokens.append((h + k) % 50_000)
    return tuple(tokens[:n])


@functools.lru_cache(maxsize=4)
def _load_tiktoken_encoding(name: str):
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            f"tokenizer='tiktoken:{name}' requested but tiktoken is not "
            f"installed. Install the optional extra: "
            f"`pip install 'GORGO[tokenizers]'`."
        ) from e
    return tiktoken.get_encoding(name)


def _content_to_tokens(
    content: str, tokenizer: str, tokens_per_char: float
) -> tuple[int, ...]:
    """Dispatch to the configured tokenizer.

    `tokenizer` is either "mock" or "tiktoken:<encoding>" (e.g.
    "tiktoken:cl100k_base"). Unknown values raise ValueError. Empty
    output is coerced to a single-token sequence so downstream code
    that assumes at least one token stays safe.
    """
    if tokenizer == "mock":
        return _mock_tokens(content, tokens_per_char)
    if tokenizer.startswith("tiktoken:"):
        enc = _load_tiktoken_encoding(tokenizer.split(":", 1)[1])
        ids = tuple(enc.encode(content))
        return ids if ids else (0,)
    raise ValueError(
        f"Unknown tokenizer {tokenizer!r}. Expected 'mock' or "
        f"'tiktoken:<encoding>'."
    )


def build_trace(
    cfg: LmsysConfig,
    params: TraceParams,
    loader: Callable[[], Iterable[SessionTurn]] | None = None,
) -> InMemoryTrace:
    """Materialize an InMemoryTrace from lmsys sessions.

    `loader` defaults to `load_sessions(cfg)` but can be injected (e.g.
    `StubLoader().iter_turns`) for tests.
    """
    if loader is None:
        def _loader():
            return load_sessions(cfg)
        source = loader_name = f"lmsys:{cfg.local_path}"
    else:
        source = loader_name = "lmsys:injected-loader"
        _loader = loader  # type: ignore[assignment]

    rng = random.Random(params.seed)
    t = 0.0
    reqs: list[Request] = []
    idx = 0
    for turn in _loader():
        if turn.role != "user":
            continue
        t += rng.expovariate(max(1e-9, params.arrival_rate_qps))
        tokens = _content_to_tokens(
            turn.content, params.tokenizer, params.tokens_per_char
        )
        reqs.append(
            Request(
                request_id=f"lm{idx:08d}",
                session_id=turn.conversation_id,
                arrival_ts=t,
                prompt_tokens=tokens,
                max_output_tokens=params.max_output_tokens,
                prefix_key=None,
                metadata={"turn_idx": turn.turn_idx, "lang": turn.language},
            )
        )
        idx += 1
        if cfg.max_conversations is not None and idx >= cfg.max_conversations * cfg.max_turns:
            break
    return InMemoryTrace(requests=reqs, source=source)
