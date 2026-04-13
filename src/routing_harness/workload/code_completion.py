"""Code-completion workload adapter.

Code-completion traces (HumanEval, MBPP, public CodeContests) look very
different from chat workloads:

  - Each task is independent (no multi-turn accumulation).
  - Prompts are code/English-code mix — cl100k tokenizes these at a
    noticeably different tokens/char ratio than conversational prose.
  - A deployed code-completion endpoint typically wraps every prompt in
    a constant instruction prefix, so KV prefix reuse across requests is
    dominated by that constant, not by cross-task similarity.

We model the shared-prefix-via-instruction pattern explicitly: an
optional `instruction_prefix` is prepended to every task prompt before
tokenization, which matches production code assistants and gives
prefix-aware routers something realistic to reuse.

Accepted record schemas (JSONL, one task per line):
  HumanEval-style:
    {"task_id": "HumanEval/0", "prompt": "...", "language": "python"?}
  MBPP-style:
    {"task_id": 11, "text": "Write a function...", "code": "..."}

The adapter reads `prompt` first, then falls back to `text`. Any
additional fields are ignored. As with the other adapters, NO network
calls are made — the user supplies a local JSONL path.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..core import Request
from .lmsys import _content_to_tokens
from .trace import InMemoryTrace


@dataclass(frozen=True)
class CodeCompletionConfig:
    local_path: str
    max_tasks: int | None = None
    # Prepended to every task prompt before tokenization. Emulates the
    # static "system" template most code-completion endpoints wrap
    # around each request — this is the dominant source of cross-request
    # prefix reuse in real deployments.
    instruction_prefix: str = ""
    language_filter: tuple[str, ...] = ()  # empty => no filter
    seed: int = 0


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    language: str


@dataclass(frozen=True)
class TraceParams:
    arrival_rate_qps: float
    tokens_per_char: float = 0.25  # only used when tokenizer == "mock"
    max_output_tokens: int = 256
    seed: int = 0
    # "mock" or "tiktoken:<encoding>". Same semantics as the lmsys adapter.
    tokenizer: str = "mock"


class StubLoader:
    """Deterministic synthetic loader for tests.

    Emits `n_tasks` code-task records that look like short HumanEval
    prompts. No shared prefix is embedded in the prompts themselves —
    cross-task reuse is expected to come from
    `CodeCompletionConfig.instruction_prefix`.
    """

    def __init__(self, n_tasks: int = 4, seed: int = 0) -> None:
        self.n_tasks = n_tasks
        self.seed = seed

    def iter_tasks(self) -> Iterator[CodeTask]:
        rng = random.Random(self.seed)
        for i in range(self.n_tasks):
            yield CodeTask(
                task_id=f"STUB/{i}",
                prompt=(
                    f"def solve_{i}(x):\n"
                    f"    # target {rng.random():.6f}\n"
                    f"    return x\n"
                ),
                language="python",
            )


def load_tasks(cfg: CodeCompletionConfig) -> Iterator[CodeTask]:
    """Stream `CodeTask` records from a local JSONL code-completion dump."""
    path = Path(cfg.local_path)
    if not path.exists():
        raise FileNotFoundError(
            f"code-completion dataset not found at {path}. Point "
            f"`local_path` at a HumanEval/MBPP/CodeContests JSONL "
            f"downloaded and unpacked manually."
        )
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt")
            if prompt is None:
                prompt = rec.get("text", "")
            language = rec.get("language", "python")
            if cfg.language_filter and language not in cfg.language_filter:
                continue
            tid = str(rec.get("task_id", rec.get("id", "")))
            yield CodeTask(task_id=tid, prompt=prompt, language=language)


def build_trace(
    cfg: CodeCompletionConfig,
    params: TraceParams,
    loader: Callable[[], Iterable[CodeTask]] | None = None,
) -> InMemoryTrace:
    """Materialize an InMemoryTrace from code-completion tasks.

    One Request per task. Each task is its own `session_id` (tasks are
    independent — no multi-turn conversation structure).
    """
    if loader is None:
        def _loader():
            return load_tasks(cfg)
        source = f"code_completion:{cfg.local_path}"
    else:
        source = "code_completion:injected-loader"
        _loader = loader  # type: ignore[assignment]

    rng = random.Random(params.seed)
    t = 0.0
    reqs: list[Request] = []
    idx = 0
    for task in _loader():
        t += rng.expovariate(max(1e-9, params.arrival_rate_qps))
        content = cfg.instruction_prefix + task.prompt
        tokens = _content_to_tokens(
            content, params.tokenizer, params.tokens_per_char
        )
        reqs.append(
            Request(
                request_id=f"cc{idx:08d}",
                session_id=task.task_id or f"task_{idx}",
                arrival_ts=t,
                prompt_tokens=tokens,
                max_output_tokens=params.max_output_tokens,
                prefix_key=None,
                metadata={"language": task.language, "task_id": task.task_id},
            )
        )
        idx += 1
        if cfg.max_tasks is not None and idx >= cfg.max_tasks:
            break
    return InMemoryTrace(requests=reqs, source=source)
