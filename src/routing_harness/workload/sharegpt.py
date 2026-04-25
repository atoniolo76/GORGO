"""ShareGPT JSONL adapter.

ShareGPT is a community dump of user/assistant turns captured from public
chat logs. Prefix-reuse distributions look materially different from
lmsys-chat-1m: conversations are longer on average, system prompts are
more varied, and code fragments show up in-line. We care about that
variation because the core research question is how well KV-aware
routing generalizes across workload shapes.

Schema assumed (one conversation per JSONL line):
  {
    "id": str,
    "conversations": [
      {"from": "human"|"gpt"|"system", "value": str},
      ...
    ]
  }

Like the lmsys adapter, this adapter does NOT download anything. The
user provides a local JSONL path.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..core import Request
from .lmsys import _content_to_tokens
from .trace import InMemoryTrace


@dataclass(frozen=True)
class ShareGPTConfig:
    local_path: str
    max_conversations: int | None = None
    min_turns: int = 1
    max_turns: int = 32
    seed: int = 0


@dataclass(frozen=True)
class SessionTurn:
    conversation_id: str
    turn_idx: int
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass(frozen=True)
class TraceParams:
    arrival_rate_qps: float
    tokens_per_char: float = 0.25  # only used when tokenizer == "mock"
    max_output_tokens: int = 256
    seed: int = 0
    # "mock" or "tiktoken:<encoding>". Same semantics as the lmsys adapter.
    tokenizer: str = "mock"


# ShareGPT's role vocabulary maps onto chat-completion roles.
_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


class StubLoader:
    """Deterministic synthetic loader for tests.

    Emits `n_convs` conversations, each with `turns_per_conv` alternating
    human/gpt turns, sharing a common system message so KV prefix reuse
    is non-trivial.
    """

    def __init__(self, n_convs: int = 4, turns_per_conv: int = 3, seed: int = 0) -> None:
        self.n_convs = n_convs
        self.turns_per_conv = turns_per_conv
        self.seed = seed

    def iter_turns(self) -> Iterator[SessionTurn]:
        rng = random.Random(self.seed)
        system_prompt = "You are ShareGPT. Be concise and accurate. " * 6
        for c in range(self.n_convs):
            cid = f"sg_{c:04d}"
            for t in range(self.turns_per_conv):
                yield SessionTurn(
                    conversation_id=cid,
                    turn_idx=t,
                    role="user" if t % 2 == 0 else "assistant",
                    content=system_prompt + f" turn {t} payload {rng.random():.6f}",
                )


def load_sessions(cfg: ShareGPTConfig) -> Iterator[SessionTurn]:
    """Stream SessionTurn records from a local ShareGPT JSONL dump.

    Only user turns are surfaced as requests in `build_trace`, but all
    roles are yielded here so downstream code can reconstruct context if
    it wants to.
    """
    path = Path(cfg.local_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ShareGPT dataset not found at {path}. Download the JSONL "
            f"dump (e.g. anon8231489123/ShareGPT_Vicuna_unfiltered on "
            f"HuggingFace) and pass the local path via config."
        )
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            convo = rec.get("conversations", [])
            if not (cfg.min_turns <= len(convo) <= cfg.max_turns):
                continue
            cid = str(rec.get("id", ""))
            for t, turn in enumerate(convo):
                raw_role = turn.get("from", "")
                role = _ROLE_MAP.get(raw_role, raw_role or "user")
                yield SessionTurn(
                    conversation_id=cid,
                    turn_idx=t,
                    role=role,
                    content=turn.get("value", ""),
                )


def build_trace(
    cfg: ShareGPTConfig,
    params: TraceParams,
    loader: Callable[[], Iterable[SessionTurn]] | None = None,
) -> InMemoryTrace:
    """Materialize an InMemoryTrace from ShareGPT sessions.

    One Request per user turn. `loader` defaults to `load_sessions(cfg)`
    but can be injected (e.g. `StubLoader().iter_turns`) for tests.
    """
    if loader is None:
        def _loader():
            return load_sessions(cfg)
        source = f"sharegpt:{cfg.local_path}"
    else:
        source = "sharegpt:injected-loader"
        _loader = loader  # type: ignore[assignment]

    rng = random.Random(params.seed)
    t = 0.0
    reqs: list[Request] = []
    idx = 0
    seen_convs: set[str] = set()
    for turn in _loader():
        if turn.role != "user":
            continue
        if (
            cfg.max_conversations is not None
            and turn.conversation_id not in seen_convs
            and len(seen_convs) >= cfg.max_conversations
        ):
            break
        seen_convs.add(turn.conversation_id)
        t += rng.expovariate(max(1e-9, params.arrival_rate_qps))
        tokens = _content_to_tokens(
            turn.content, params.tokenizer, params.tokens_per_char
        )
        reqs.append(
            Request(
                request_id=f"sg{idx:08d}",
                session_id=turn.conversation_id,
                arrival_ts=t,
                prompt_tokens=tokens,
                max_output_tokens=params.max_output_tokens,
                prefix_key=None,
                metadata={"turn_idx": turn.turn_idx},
            )
        )
        idx += 1
    return InMemoryTrace(
        requests=reqs,
        source=source,
        params={"cfg": asdict(cfg), "trace": asdict(params)},
    )
