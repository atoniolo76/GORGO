"""Massively parallel Mooncake trace curation sweeps over production windows.

The sweep is intentionally just orchestration around
``build_mooncake_trace``: each window/config pair is independent, so we use
Modal's input mapping to fan out work and then reduce the sidecar summaries
into a manifest ranked by overlap/diversity.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import modal

from app import app, completions_volume
from data_processing.build_mooncake_trace import build_mooncake_trace

sweep_image = modal.Image.debian_slim().add_local_python_source("app", "data_processing")
MAX_SWEEP_CONTAINERS = 32


@dataclass
class WindowSpec:
    sweep_id: str
    window_idx: int
    start_time: str
    end_time: str
    num_requests: int
    selection_mode: str
    candidate_multiplier: int
    top_token_hashes: int
    max_input_tokens: int
    max_total_tokens: int
    min_input_tokens: int
    time_scale: float
    target_duration_ms: int
    block_size: int
    include_bodies: bool
    config_slug: str
    body_output_dir: str = ""

    @property
    def stem(self) -> str:
        start = self.start_time.replace(":", "").replace("-", "").replace("+00:00", "Z")
        mode = self.selection_mode.replace("-", "_")
        return f"{self.window_idx:05d}_{start}_{mode}_top{self.top_token_hashes}"

    @property
    def output_path(self) -> str:
        if self.include_bodies:
            body_dir = self.body_output_dir or "with_bodies"
            return f"mooncake_traces/sweeps/{self.sweep_id}/{self.config_slug}/{body_dir}/{self.stem}.jsonl"
        return f"mooncake_traces/sweeps/{self.sweep_id}/{self.config_slug}/{self.stem}.jsonl"

    @property
    def sidecar_path(self) -> str:
        if self.include_bodies:
            body_dir = self.body_output_dir or "with_bodies"
            return f"mooncake_traces/sweeps/{self.sweep_id}/{self.config_slug}/{body_dir}/{self.stem}.summary.json"
        return f"mooncake_traces/sweeps/{self.sweep_id}/{self.config_slug}/{self.stem}.summary.json"


def _parse_iso(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _iso_z(d: datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_windows(start: str, end: str, window_minutes: int) -> list[tuple[str, str]]:
    """Return half-open ISO8601 windows covering ``[start, end)``."""
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    cur = _parse_iso(start)
    stop = _parse_iso(end)
    step = timedelta(minutes=window_minutes)
    out: list[tuple[str, str]] = []
    while cur < stop:
        nxt = min(cur + step, stop)
        out.append((_iso_z(cur), _iso_z(nxt)))
        cur = nxt
    return out


def _slug_part(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def _config_slug(config: dict) -> str:
    """Stable, human-readable slug plus short hash for output isolation."""
    stable = json.dumps(config, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(stable.encode()).hexdigest()[:10]
    return "_".join(
        [
            _slug_part(str(config["selection_mode"])),
            f"n{config['num_requests']}",
            f"w{config['window_minutes']}m",
            f"cand{config['candidate_multiplier']}",
            f"top{_slug_part(config['top_token_hashes_csv'])}",
            f"maxin{config['max_input_tokens']}",
            f"scale{_slug_part(str(config['time_scale']))}",
            h,
        ]
    )


@app.function(
    image=sweep_image,
    volumes={"/data": completions_volume},
    timeout=30 * 60,
    memory=1024 * 4,
    max_containers=MAX_SWEEP_CONTAINERS,
)
def build_trace_for_window(spec: dict) -> dict:
    """Wrapper around ``build_mooncake_trace`` for Modal ``map`` fanout."""
    ws = WindowSpec(**spec)
    try:
        result = build_mooncake_trace.remote(
            start_time=ws.start_time,
            num_requests=ws.num_requests,
            output_path=ws.output_path,
            end_time=ws.end_time,
            block_size=ws.block_size,
            include_bodies=ws.include_bodies,
            time_scale=ws.time_scale,
            target_duration_ms=ws.target_duration_ms,
            selection_mode=ws.selection_mode,
            candidate_multiplier=ws.candidate_multiplier,
            top_token_hashes=ws.top_token_hashes,
            max_input_tokens=ws.max_input_tokens,
            max_total_tokens=ws.max_total_tokens,
            min_input_tokens=ws.min_input_tokens,
            output_sidecar_path=ws.sidecar_path,
        )
        return {"ok": True, "spec": spec, "result": result}
    except Exception as e:
        return {"ok": False, "spec": spec, "error": f"{type(e).__name__}: {e}"}


def _rank_score(result: dict) -> float:
    if not result.get("ok"):
        return float("-inf")
    r = result.get("result") or {}
    reuse = float(r.get("block_reuse_pct") or 0.0)
    rows = int(r.get("rows") or 0)
    tokens = float(r.get("total_input_tokens") or 0.0)
    duration_ms = float(r.get("original_duration_ms") or 0.0)
    # Favor reuse first, then enough rows/tokens/duration to avoid degenerate
    # tiny traces. Keep simple because the manifest also preserves raw stats.
    return (
        reuse
        + min(rows, 10_000) / 10_000
        + min(tokens, 50_000_000) / 50_000_000
        + min(duration_ms, 3_600_000) / 3_600_000
    )


def _make_specs(
    *,
    sweep_id: str,
    start_time: str,
    end_time: str,
    window_minutes: int,
    num_requests: int,
    selection_mode: str,
    top_token_hashes_csv: str,
    candidate_multiplier: int,
    max_input_tokens: int,
    max_total_tokens: int,
    min_input_tokens: int,
    time_scale: float,
    target_duration_ms: int,
    block_size: int,
    include_bodies: bool,
    config_slug: str | None = None,
    body_output_dir: str = "",
) -> list[dict]:
    slug = config_slug or _config_slug(
        {
            "start_time": start_time,
            "end_time": end_time,
            "window_minutes": window_minutes,
            "num_requests": num_requests,
            "selection_mode": selection_mode,
            "top_token_hashes_csv": top_token_hashes_csv,
            "candidate_multiplier": candidate_multiplier,
            "max_input_tokens": max_input_tokens,
            "max_total_tokens": max_total_tokens,
            "min_input_tokens": min_input_tokens,
            "time_scale": time_scale,
            "target_duration_ms": target_duration_ms,
            "block_size": block_size,
        }
    )
    top_token_hashes_values = [
        int(x.strip()) for x in top_token_hashes_csv.split(",") if x.strip()
    ] or [0]
    windows = make_windows(start_time, end_time, window_minutes)
    specs: list[dict] = []
    for idx, (w_start, w_end) in enumerate(windows):
        for kth in top_token_hashes_values:
            specs.append(
                asdict(
                    WindowSpec(
                        sweep_id=sweep_id,
                        window_idx=idx,
                        start_time=w_start,
                        end_time=w_end,
                        num_requests=num_requests,
                        selection_mode=selection_mode,
                        candidate_multiplier=candidate_multiplier,
                        top_token_hashes=kth,
                        max_input_tokens=max_input_tokens,
                        max_total_tokens=max_total_tokens,
                        min_input_tokens=min_input_tokens,
                        time_scale=time_scale,
                        target_duration_ms=target_duration_ms,
                        block_size=block_size,
                        include_bodies=include_bodies,
                        config_slug=slug,
                        body_output_dir=body_output_dir,
                    )
                )
            )
    return specs


def _sweep_dir(sweep_id: str, config_slug: str | None = None) -> str:
    base = f"/data/mooncake_traces/sweeps/{sweep_id}"
    return os.path.join(base, config_slug) if config_slug else base


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _rank_and_manifest(
    *,
    sweep_id: str,
    start_time: str,
    end_time: str,
    window_minutes: int,
    results: list[dict],
    top_k: int,
    rebuilt: list | None = None,
    extra: dict | None = None,
) -> dict:
    ranked = sorted(results, key=_rank_score, reverse=True)
    top = ranked[:top_k]
    config_slug = None
    if results:
        config_slug = (results[0].get("spec") or {}).get("config_slug")
    manifest_path = os.path.join(_sweep_dir(sweep_id, config_slug), "manifest.json")
    manifest = {
        "sweep_id": sweep_id,
        "config_slug": config_slug,
        "start_time": start_time,
        "end_time": end_time,
        "window_minutes": window_minutes,
        "num_specs": len(results),
        "top_k": top_k,
        "ranked": ranked,
        "top": top,
        "rebuilt_with_bodies": rebuilt or [],
        **(extra or {}),
    }
    _write_json_atomic(manifest_path, manifest)
    completions_volume.commit()
    return {"manifest_path": manifest_path, "num_specs": len(results), "top": top}


@app.function(
    image=sweep_image,
    volumes={"/data": completions_volume},
    timeout=60 * 60,
    memory=1024 * 8,
    max_containers=1,
)
def sweep(
    sweep_id: str,
    start_time: str,
    end_time: str,
    window_minutes: int = 15,
    num_requests: int = 1000,
    selection_mode: str = "token-hash-filter",
    top_token_hashes_csv: str = "5,10,20,50",
    candidate_multiplier: int = 20,
    max_input_tokens: int = 0,
    max_total_tokens: int = 0,
    min_input_tokens: int = 0,
    time_scale: float = 1.0,
    target_duration_ms: int = 0,
    block_size: int = 512,
    top_k: int = 20,
    rebuild_top_k_with_bodies: bool = False,
    map_order_outputs: bool = False,
) -> dict:
    """Fan out trace builds over windows, rank sidecars, and write a manifest."""
    completions_volume.reload()
    specs = _make_specs(
        sweep_id=sweep_id,
        start_time=start_time,
        end_time=end_time,
        window_minutes=window_minutes,
        num_requests=num_requests,
        selection_mode=selection_mode,
        top_token_hashes_csv=top_token_hashes_csv,
        candidate_multiplier=candidate_multiplier,
        max_input_tokens=max_input_tokens,
        max_total_tokens=max_total_tokens,
        min_input_tokens=min_input_tokens,
        time_scale=time_scale,
        target_duration_ms=target_duration_ms,
        block_size=block_size,
        include_bodies=False,
    )

    results = list(
        build_trace_for_window.map(
            specs,
            return_exceptions=True,
            order_outputs=map_order_outputs,
        )
    )
    normalized: list[dict] = []
    for spec, result in zip(specs, results):
        if isinstance(result, Exception):
            normalized.append({"ok": False, "spec": spec, "error": repr(result)})
        else:
            normalized.append(result)

    top = sorted(normalized, key=_rank_score, reverse=True)[:top_k]

    rebuilt: list[dict] = []
    if rebuild_top_k_with_bodies:
        body_specs: list[dict] = []
        for item in top:
            if not item.get("ok"):
                continue
            spec = dict(item["spec"])
            spec["include_bodies"] = True
            body_specs.append(spec)
        rebuilt = list(
            build_trace_for_window.map(
                body_specs,
                return_exceptions=True,
                order_outputs=map_order_outputs,
            )
        )

    return _rank_and_manifest(
        sweep_id=sweep_id,
        start_time=start_time,
        end_time=end_time,
        window_minutes=window_minutes,
        results=normalized,
        top_k=top_k,
        rebuilt=[r if not isinstance(r, Exception) else repr(r) for r in rebuilt],
        extra={"mode": "blocking-map"},
    )


@app.function(
    image=sweep_image,
    volumes={"/data": completions_volume},
    timeout=30 * 60,
    memory=1024 * 4,
    max_containers=1,
)
def launch_sweep(
    sweep_id: str,
    start_time: str,
    end_time: str,
    window_minutes: int = 15,
    num_requests: int = 1000,
    selection_mode: str = "token-hash-filter",
    top_token_hashes_csv: str = "5,10,20,50",
    candidate_multiplier: int = 20,
    max_input_tokens: int = 0,
    max_total_tokens: int = 0,
    min_input_tokens: int = 0,
    time_scale: float = 1.0,
    target_duration_ms: int = 0,
    block_size: int = 512,
) -> dict:
    """Spawn independent window jobs and persist their call IDs for later collection."""
    completions_volume.reload()
    specs = _make_specs(
        sweep_id=sweep_id,
        start_time=start_time,
        end_time=end_time,
        window_minutes=window_minutes,
        num_requests=num_requests,
        selection_mode=selection_mode,
        top_token_hashes_csv=top_token_hashes_csv,
        candidate_multiplier=candidate_multiplier,
        max_input_tokens=max_input_tokens,
        max_total_tokens=max_total_tokens,
        min_input_tokens=min_input_tokens,
        time_scale=time_scale,
        target_duration_ms=target_duration_ms,
        block_size=block_size,
        include_bodies=False,
    )
    calls = []
    for spec in specs:
        call = build_trace_for_window.spawn(spec)
        calls.append({"call_id": call.object_id, "spec": spec})
    config_slug = specs[0]["config_slug"] if specs else "empty"
    launch_manifest_path = os.path.join(_sweep_dir(sweep_id, config_slug), "launch_manifest.json")
    launch_manifest = {
        "sweep_id": sweep_id,
        "config_slug": config_slug,
        "start_time": start_time,
        "end_time": end_time,
        "window_minutes": window_minutes,
        "num_specs": len(specs),
        "calls": calls,
    }
    _write_json_atomic(launch_manifest_path, launch_manifest)
    completions_volume.commit()
    return {
        "launch_manifest_path": launch_manifest_path,
        "config_slug": config_slug,
        "num_specs": len(specs),
    }


@app.function(
    image=sweep_image,
    volumes={"/data": completions_volume},
    timeout=60 * 60,
    memory=1024 * 8,
    max_containers=1,
)
def collect_sweep(
    sweep_id: str,
    config_slug: str = "",
    top_k: int = 20,
    wait: bool = False,
    rebuild_top_k_with_bodies: bool = False,
) -> dict:
    """Collect spawned window jobs by call id and write the ranked manifest.

    If the sweep was run via the blocking ``sweep_main``/``.map`` path, there
    is no ``launch_manifest.json``. In that case, fall back to the existing
    ``manifest.json`` and optionally rebuild its top traces with bodies.
    """
    completions_volume.reload()
    if not config_slug:
        raise ValueError("config_slug is required; use the value returned by launch_sweep_main")
    sweep_dir = _sweep_dir(sweep_id, config_slug)
    launch_manifest_path = os.path.join(sweep_dir, "launch_manifest.json")
    manifest_path = os.path.join(sweep_dir, "manifest.json")

    if os.path.exists(launch_manifest_path):
        with open(launch_manifest_path) as f:
            launch = json.load(f)

        results: list[dict] = []
        pending: list[dict] = []
        for item in launch["calls"]:
            call = modal.FunctionCall.from_id(item["call_id"])
            try:
                result = call.get(timeout=None if wait else 0)
            except TimeoutError:
                pending.append(item)
                continue
            except Exception as e:
                result = {"ok": False, "spec": item["spec"], "error": f"{type(e).__name__}: {e}"}
            results.append(result)
        start_time = launch["start_time"]
        end_time = launch["end_time"]
        window_minutes = launch["window_minutes"]
        mode = "spawn-collect"
    elif os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        results = manifest.get("ranked") or []
        pending = []
        start_time = manifest.get("start_time")
        end_time = manifest.get("end_time")
        window_minutes = manifest.get("window_minutes")
        mode = "manifest-rebuild"
    else:
        raise FileNotFoundError(
            f"neither {launch_manifest_path!r} nor {manifest_path!r} exists; "
            "run launch_sweep_main or sweep_main first"
        )

    rebuilt: list = []
    if rebuild_top_k_with_bodies and not pending:
        top = sorted(results, key=_rank_score, reverse=True)[:top_k]
        body_calls = []
        for item in top:
            if not item.get("ok"):
                continue
            spec = dict(item["spec"])
            spec["include_bodies"] = True
            spec["body_output_dir"] = "with_bodies"
            body_calls.append(build_trace_for_window.spawn(spec))
        for call in body_calls:
            try:
                rebuilt.append(call.get(timeout=None if wait else 0))
            except TimeoutError:
                rebuilt.append(
                    {"ok": False, "error": "body rebuild pending", "call_id": call.object_id}
                )
            except Exception as e:
                rebuilt.append({"ok": False, "error": f"{type(e).__name__}: {e}"})

    summary = _rank_and_manifest(
        sweep_id=sweep_id,
        start_time=start_time,
        end_time=end_time,
        window_minutes=window_minutes,
        results=results,
        top_k=top_k,
        rebuilt=rebuilt,
        extra={
            "mode": mode,
            "pending": pending,
            "completed": len(results),
        },
    )
    summary["pending"] = len(pending)
    summary["completed"] = len(results)
    return summary


@app.local_entrypoint()
def sweep_main(
    sweep_id: str,
    start_time: str,
    end_time: str,
    window_minutes: int = 15,
    num_requests: int = 1000,
    selection_mode: str = "token-hash-filter",
    top_token_hashes_csv: str = "5,10,20,50",
    candidate_multiplier: int = 20,
    max_input_tokens: int = 0,
    max_total_tokens: int = 0,
    min_input_tokens: int = 0,
    time_scale: float = 1.0,
    target_duration_ms: int = 0,
    block_size: int = 512,
    top_k: int = 20,
    rebuild_top_k_with_bodies: bool = False,
    map_order_outputs: bool = False,
):
    result = sweep.remote(
        sweep_id=sweep_id,
        start_time=start_time,
        end_time=end_time,
        window_minutes=window_minutes,
        num_requests=num_requests,
        selection_mode=selection_mode,
        top_token_hashes_csv=top_token_hashes_csv,
        candidate_multiplier=candidate_multiplier,
        max_input_tokens=max_input_tokens,
        max_total_tokens=max_total_tokens,
        min_input_tokens=min_input_tokens,
        time_scale=time_scale,
        target_duration_ms=target_duration_ms,
        block_size=block_size,
        top_k=top_k,
        rebuild_top_k_with_bodies=rebuild_top_k_with_bodies,
        map_order_outputs=map_order_outputs,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def launch_sweep_main(
    sweep_id: str,
    start_time: str,
    end_time: str,
    window_minutes: int = 15,
    num_requests: int = 1000,
    selection_mode: str = "token-hash-filter",
    top_token_hashes_csv: str = "5,10,20,50",
    candidate_multiplier: int = 20,
    max_input_tokens: int = 0,
    max_total_tokens: int = 0,
    min_input_tokens: int = 0,
    time_scale: float = 1.0,
    target_duration_ms: int = 0,
    block_size: int = 512,
):
    """Spawn a durable sweep and write ``launch_manifest.json`` with call IDs."""
    result = launch_sweep.remote(
        sweep_id=sweep_id,
        start_time=start_time,
        end_time=end_time,
        window_minutes=window_minutes,
        num_requests=num_requests,
        selection_mode=selection_mode,
        top_token_hashes_csv=top_token_hashes_csv,
        candidate_multiplier=candidate_multiplier,
        max_input_tokens=max_input_tokens,
        max_total_tokens=max_total_tokens,
        min_input_tokens=min_input_tokens,
        time_scale=time_scale,
        target_duration_ms=target_duration_ms,
        block_size=block_size,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def collect_sweep_main(
    sweep_id: str,
    config_slug: str = "",
    top_k: int = 20,
    wait: bool = False,
    rebuild_top_k_with_bodies: bool = False,
):
    """Collect spawned sweep calls and write/update ``manifest.json``."""
    result = collect_sweep.remote(
        sweep_id=sweep_id,
        config_slug=config_slug,
        top_k=top_k,
        wait=wait,
        rebuild_top_k_with_bodies=rebuild_top_k_with_bodies,
    )
    print(json.dumps(result, indent=2))
