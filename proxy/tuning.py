"""CLI/TUI client for proxy-embedded batch tuning.

The tuning loop itself lives in ``proxy/modal_proxy.py`` and runs inside the
proxy container, replaying workload traffic against ``http://127.0.0.1:8000``.
This module only collects knobs, starts a run via ``POST /tuning/start``, and
optionally polls ``GET /tuning/status`` until completion.
"""

from __future__ import annotations

import time

import modal

from app import app

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
HYPERPARAM_RANGES: dict[str, tuple[float, float]] = {
    "t_prefill": (1e-5, 1.0),
    "queued_tokens_weight": (1e-5, 1.0),
}
METRIC_CHOICES = [
    "neg_avg_itl",
    "neg_p95_e2e",
    "neg_p95_ttft",
    "neg_p99_ttft",
    "output_throughput",
    "request_throughput",
    "total_throughput",
]


def _parse_stream_flag(stream: str) -> bool | None:
    s = stream.strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise SystemExit(f"invalid stream={stream!r}; expected true/false")


def _dispatch_tune(
    *,
    proxy_url: str,
    source: str,
    data_path: str,
    arrival_mode: str,
    time_scale: float,
    start_time: str,
    end_time: str,
    offset: int,
    num_requests: int,
    concurrency: int,
    model: str,
    stream: str,
    max_tokens: int,
    max_input_tokens: int,
    metric: str,
    max_steps: int,
    sigma: float,
    sigma_min: float,
    seed: int,
    relative_tolerance: float,
    output_dir: str,
    t_prefill_min: float,
    t_prefill_max: float,
    queued_tokens_weight_min: float,
    queued_tokens_weight_max: float,
    wait: bool = True,
    poll_interval_seconds: float = 5.0,
):
    import httpx

    base_url = proxy_url.rstrip("/")
    payload = {
        "source": source or "glm5",
        "data_path": data_path or None,
        "arrival_mode": arrival_mode or "bounded",
        "time_scale": time_scale,
        "start_time": start_time or None,
        "end_time": end_time or None,
        "offset": offset,
        "num_requests": num_requests or None,
        "concurrency": concurrency,
        "model": model or None,
        "stream": _parse_stream_flag(stream),
        "max_tokens": max_tokens or None,
        "max_input_tokens": max_input_tokens,
        "metric": metric,
        "max_steps": max_steps,
        "sigma": sigma,
        "sigma_min": sigma_min,
        "seed": seed,
        "relative_tolerance": relative_tolerance,
        "output_dir": output_dir or None,
        "t_prefill_min": t_prefill_min,
        "t_prefill_max": t_prefill_max,
        "queued_tokens_weight_min": queued_tokens_weight_min,
        "queued_tokens_weight_max": queued_tokens_weight_max,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(30.0)) as client:
        r = client.post("/tuning/start", json=payload)
        if r.status_code >= 400:
            raise SystemExit(f"start tuning: HTTP {r.status_code}: {r.text}")
        doc = r.json()
        state_doc = doc.get("batch_tuning") or doc
        print(
            f"[tune-cli] started run_id={doc.get('run_id') or state_doc.get('run_id')} "
            f"status={state_doc.get('status')}"
        )
        if not wait:
            return doc

        last_line = None
        while True:
            s = client.get("/tuning/status")
            if s.status_code >= 400:
                raise SystemExit(f"poll tuning: HTTP {s.status_code}: {s.text}")
            status_doc = s.json().get("batch_tuning") or {}
            history = status_doc.get("history") or []
            latest = history[-1] if history else None
            line = (
                f"[tune-cli] status={status_doc.get('status')} "
                f"trial={status_doc.get('trial')} "
                f"best_score={status_doc.get('best_score')} "
                f"best_params={status_doc.get('best_params')} "
                f"output={status_doc.get('output_path')}"
            )
            if latest:
                line += f" latest_score={latest.get('score')}"
            if line != last_line:
                print(line, flush=True)
                last_line = line
            if status_doc.get("status") in ("succeeded", "failed", "cancelled"):
                if status_doc.get("error"):
                    print(f"[tune-cli] error={status_doc['error']}")
                return status_doc
            time.sleep(poll_interval_seconds)


_TUNE_TUI_SPEC: list[tuple[str, list[dict]]] = [
    (
        "Workload (mirrors proxy/workload.py)",
        [
            {
                "name": "source",
                "kind": "choice",
                "default": "glm5",
                "help": "Workload source",
                "choices": ["glm5", "hf", "mooncake"],
            },
            {
                "name": "data_path",
                "kind": "str",
                "default": "",
                "help": "Dataset/trace path; required for source=mooncake",
            },
            {
                "name": "arrival_mode",
                "kind": "choice",
                "default": "bounded",
                "help": "bounded or open-loop",
                "choices": ["bounded", "open-loop"],
            },
            {
                "name": "time_scale",
                "kind": "float",
                "default": 1.0,
                "help": "Replay-time timestamp multiplier for Mooncake traces",
            },
            {
                "name": "start_time",
                "kind": "str",
                "default": "",
                "help": "YYYY-MM-DD start of replay window; empty = use first row",
            },
            {
                "name": "end_time",
                "kind": "str",
                "default": "",
                "help": "YYYY-MM-DD end of replay window; empty = unbounded",
            },
            {"name": "offset", "kind": "int", "default": 0, "help": "Skip N usable rows"},
            {
                "name": "num_requests",
                "kind": "int",
                "default": 0,
                "help": "0 = full window",
            },
            {
                "name": "concurrency",
                "kind": "int",
                "default": 16,
                "help": "Max in-flight requests per trial",
            },
            {
                "name": "model",
                "kind": "str",
                "default": DEFAULT_MODEL,
                "help": "Model name override sent to upstream",
            },
            {
                "name": "stream",
                "kind": "str",
                "default": "",
                "help": "true/false/empty -- empty uses replay default",
            },
            {"name": "max_tokens", "kind": "int", "default": 0, "help": "0 = use replay default"},
            {
                "name": "max_input_tokens",
                "kind": "int",
                "default": 0,
                "help": "0=auto from /get_server_info, <0=disable, >0=explicit cap",
            },
        ],
    ),
    (
        "Tuner",
        [
            {
                "name": "metric",
                "kind": "choice",
                "default": "output_throughput",
                "help": "Score function to maximize",
                "choices": METRIC_CHOICES,
            },
            {"name": "max_steps", "kind": "int", "default": 16, "help": "Trials after baseline"},
            {
                "name": "relative_tolerance",
                "kind": "float",
                "default": 0.005,
                "help": "Improvement fraction to count as real",
            },
            {
                "name": "sigma",
                "kind": "float",
                "default": 0.5,
                "help": "gaussian-es initial log-space sigma",
            },
            {"name": "sigma_min", "kind": "float", "default": 0.02, "help": "gaussian-es floor"},
            {"name": "seed", "kind": "int", "default": -1, "help": "-1 = nondeterministic"},
        ],
    ),
    (
        "Hyperparameter search ranges (log-space; lower bound > 0)",
        [
            {
                "name": "t_prefill_min",
                "kind": "float",
                "default": HYPERPARAM_RANGES["t_prefill"][0],
                "help": "Lower bound for t_prefill",
            },
            {
                "name": "t_prefill_max",
                "kind": "float",
                "default": HYPERPARAM_RANGES["t_prefill"][1],
                "help": "Upper bound for t_prefill",
            },
            {
                "name": "queued_tokens_weight_min",
                "kind": "float",
                "default": HYPERPARAM_RANGES["queued_tokens_weight"][0],
                "help": "Lower bound for queued_tokens_weight",
            },
            {
                "name": "queued_tokens_weight_max",
                "kind": "float",
                "default": HYPERPARAM_RANGES["queued_tokens_weight"][1],
                "help": "Upper bound for queued_tokens_weight",
            },
        ],
    ),
    (
        "Output",
        [
            {
                "name": "output_dir",
                "kind": "str",
                "default": "",
                "help": "Empty = /results/tune_<UTC-timestamp>",
            },
        ],
    ),
]


def _coerce_value(kind: str, raw: str, default):
    if raw == "":
        return default
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    return raw


def _prompt_field_plain(field: dict) -> object:
    name = field["name"]
    default = field["default"]
    label = f"  {name}"
    if field.get("help"):
        label += f"  ({field['help']})"
    if field.get("choices"):
        label += f"  [{'|'.join(field['choices'])}]"
    label += f" [{default!r}]: "
    while True:
        raw = input(label).strip()
        try:
            value = _coerce_value(field["kind"], raw, default)
        except ValueError as e:
            print(f"    invalid: {e}")
            continue
        if field.get("choices") and value not in field["choices"]:
            print(f"    must be one of {field['choices']}")
            continue
        return value


def _print_config_table_plain(chosen: dict) -> None:
    name_width = max(len(k) for k in chosen)
    print()
    print("Final configuration:")
    print("-" * (name_width + 30))
    for k, v in chosen.items():
        print(f"  {k.ljust(name_width)}  {v!r}")
    print("-" * (name_width + 30))


def _run_tui(initial_proxy_url: str = "") -> dict:
    try:
        from rich.console import Console
        from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
        from rich.table import Table

        rich_available = True
    except ImportError:
        rich_available = False

    chosen: dict = {}
    if rich_available:
        console = Console()
        console.rule("[bold cyan]GORGO online tuner[/]")
        proxy_url = initial_proxy_url or Prompt.ask(
            "[bold]proxy_url[/] [dim](base URL of running proxy)[/]"
        )
        chosen["proxy_url"] = proxy_url.strip()
        for section, fields in _TUNE_TUI_SPEC:
            console.print(f"\n[bold magenta]{section}[/]")
            for f in fields:
                label = f"  [cyan]{f['name']}[/]"
                if f.get("help"):
                    label += f" [dim]{f['help']}[/]"
                if f["kind"] == "int":
                    chosen[f["name"]] = IntPrompt.ask(
                        label, default=int(f["default"]), console=console
                    )
                elif f["kind"] == "float":
                    chosen[f["name"]] = FloatPrompt.ask(
                        label, default=float(f["default"]), console=console
                    )
                elif f["kind"] == "choice":
                    chosen[f["name"]] = Prompt.ask(
                        label,
                        default=str(f["default"]),
                        choices=f["choices"],
                        console=console,
                    )
                else:
                    chosen[f["name"]] = Prompt.ask(
                        label, default=str(f["default"]), console=console
                    )
        table = Table(title="Final configuration", show_lines=False)
        table.add_column("param", style="cyan", no_wrap=True)
        table.add_column("value", style="white")
        for k, v in chosen.items():
            table.add_row(k, repr(v))
        console.print()
        console.print(table)
        if not Confirm.ask("Launch this run?", default=True):
            console.print("[yellow]aborted[/]")
            raise SystemExit(0)
    else:
        print("=== GORGO online tuner ===")
        proxy_url = initial_proxy_url or input("  proxy_url (base URL of running proxy): ")
        chosen["proxy_url"] = proxy_url.strip()
        for section, fields in _TUNE_TUI_SPEC:
            print(f"\n[{section}]")
            for f in fields:
                chosen[f["name"]] = _prompt_field_plain(f)
        _print_config_table_plain(chosen)
        confirm = input("Launch this run? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("aborted")
            raise SystemExit(0)
    return chosen


@app.local_entrypoint()
def tune_cli(
    proxy_url: str,
    source: str = "glm5",
    data_path: str = "",
    arrival_mode: str = "bounded",
    time_scale: float = 1.0,
    start_time: str = "",
    end_time: str = "",
    offset: int = 0,
    num_requests: int = 0,
    concurrency: int = 16,
    model: str = DEFAULT_MODEL,
    stream: str = "",
    max_tokens: int = 0,
    max_input_tokens: int = 0,
    metric: str = "output_throughput",
    algorithm: str = "gaussian-es",
    max_steps: int = 16,
    sigma: float = 0.5,
    sigma_min: float = 0.02,
    seed: int = -1,
    relative_tolerance: float = 0.005,
    output_dir: str = "",
    t_prefill_min: float = HYPERPARAM_RANGES["t_prefill"][0],
    t_prefill_max: float = HYPERPARAM_RANGES["t_prefill"][1],
    queued_tokens_weight_min: float = HYPERPARAM_RANGES["queued_tokens_weight"][0],
    queued_tokens_weight_max: float = HYPERPARAM_RANGES["queued_tokens_weight"][1],
    no_wait: bool = False,
    poll_interval_seconds: float = 5.0,
):
    """Start a proxy-embedded tuning run and optionally poll until completion."""
    _dispatch_tune(
        proxy_url=proxy_url,
        source=source,
        data_path=data_path,
        arrival_mode=arrival_mode,
        time_scale=time_scale,
        start_time=start_time,
        end_time=end_time,
        offset=offset,
        num_requests=num_requests,
        concurrency=concurrency,
        model=model,
        stream=stream,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        metric=metric,
        max_steps=max_steps,
        sigma=sigma,
        sigma_min=sigma_min,
        seed=seed,
        relative_tolerance=relative_tolerance,
        output_dir=output_dir,
        t_prefill_min=t_prefill_min,
        t_prefill_max=t_prefill_max,
        queued_tokens_weight_min=queued_tokens_weight_min,
        queued_tokens_weight_max=queued_tokens_weight_max,
        wait=not no_wait,
        poll_interval_seconds=poll_interval_seconds,
    )


def _suppress_modal_status_spinner() -> None:
    try:
        from modal._output.manager import OutputManager

        mgr = OutputManager.get()
        stop = getattr(mgr, "stop_status_spinner", None)
        if callable(stop):
            stop()
    except Exception:
        pass


@app.local_entrypoint()
def tune_interactive(proxy_url: str = ""):
    """Interactive TUI that starts a proxy-embedded tuning run."""
    _suppress_modal_status_spinner()
    chosen = _run_tui(initial_proxy_url=proxy_url)
    _dispatch_tune(**chosen)
