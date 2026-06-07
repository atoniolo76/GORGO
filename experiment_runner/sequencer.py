"""Multi-phase experiment sequencer.

Runs a tuning phase followed by N eval phases, each as a separate
``run_policy_matrix_experiment`` invocation with its own fleet. Learned
weights are persisted to the bench-results volume between phases so they
survive process restarts.

This avoids container collisions between phases: each phase spins up
fresh engines/proxies and tears them down before the next phase starts.

Usage — full run (tune + 2 evals):
    modal run --detach --env=alessio-dev experiment_runner/sequencer.py::main \\
        --tuning-spec-path specs/c64/tuning/policy_matrix_c64_tuning_p95ttft.json \\
        --tuning-manifest-path specs/c64/manifests/manifest_glm5_apr2_0030_0100.json \\
        --eval-spec-path specs/c64/eval/policy_matrix_c64_eval_p95ttft.json \\
        --eval-manifest-paths "specs/c64/manifests/manifest_glm5_apr2_0100_0130.json,specs/c64/manifests/manifest_glm5_apr2_1230_1300.json" \\
        --experiment-id glm5_c64_tuning_p95ttft_v6 \\
        --output-dir /results/policy_matrix_sweep/c64/glm5_c64_tuning_p95ttft_v6

Resume evals only (tuning already completed, weights on volume):
    modal run --detach --env=alessio-dev experiment_runner/sequencer.py::main \\
        --eval-spec-path specs/c64/eval/policy_matrix_c64_eval_p95ttft.json \\
        --eval-manifest-paths "specs/c64/manifests/manifest_glm5_apr2_1230_1300.json" \\
        --experiment-id glm5_c64_tuning_p95ttft_v6 \\
        --output-dir /results/policy_matrix_sweep/c64/glm5_c64_tuning_p95ttft_v6 \\
        --skip-tuning
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import modal

from app import app, bench_results_volume
from experiment_runner.policy_matrix_app import (
    _capture_environment,
    _validate_spec,
    run_policy_matrix_experiment,
)

WEIGHTS_FILENAME = "learned_weights.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _weights_volume_path(output_dir: str) -> str:
    return f"{output_dir}/{WEIGHTS_FILENAME}"


def _save_weights_to_volume(weights: dict, output_dir: str) -> None:
    """Persist learned weights to the bench-results volume."""
    vpath = _weights_volume_path(output_dir)
    p = Path(vpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(weights, indent=2))
    bench_results_volume.commit()
    print(f"[sequencer] saved weights to volume: {vpath}", flush=True)


def _load_weights_from_volume(output_dir: str) -> dict | None:
    """Load previously saved weights from the bench-results volume."""
    vpath = _weights_volume_path(output_dir)
    p = Path(vpath)
    if not p.exists():
        # Volume might need a reload to see files written by other containers
        bench_results_volume.reload()
        if not p.exists():
            return None
    try:
        weights = json.loads(p.read_text())
        print(f"[sequencer] loaded weights from volume: {vpath}", flush=True)
        return weights
    except (json.JSONDecodeError, OSError) as e:
        print(f"[sequencer][warn] failed to load {vpath}: {e}", flush=True)
        return None


SEQUENCER_IMAGE = (
    modal.Image.debian_slim()
    .pip_install("httpx")
    .add_local_python_source(
        "app", "engine", "experiment_runner", "proxy", "policy", "scripts", "utils"
    )
)


@app.function(
    image=SEQUENCER_IMAGE,
    volumes={"/results": bench_results_volume},
    timeout=4 * 3600,
)
def run_sequenced_experiment(
    *,
    tuning_spec: dict | None,
    tuning_manifest: dict | None,
    eval_spec: dict | None,
    eval_manifests: list[dict],
    experiment_id: str,
    output_dir: str,
    start_index: int,
    top_k: int,
    cooldown_seconds: float,
    skip_tuning: bool,
    environment: dict,
) -> dict:
    """Run tuning + N evals as separate fleet invocations.

    Runs entirely on Modal so it survives local disconnects.
    """
    sequencer_manifest = {
        "experiment_id": experiment_id,
        "started_utc": _utc_now(),
        "phases": [],
    }

    learned = None

    # ----------------------------------------------------------------
    # Phase 0: Tuning (or load weights from volume)
    # ----------------------------------------------------------------
    if skip_tuning:
        print(f"\n[sequencer] skipping tuning, loading weights from volume...", flush=True)
        learned = _load_weights_from_volume(output_dir)
        if not learned:
            print("[sequencer][ERROR] --skip-tuning but no weights found on volume", flush=True)
            _write_sequencer_manifest(sequencer_manifest, output_dir)
            return sequencer_manifest
        print(f"[sequencer] resumed with weights: {learned}", flush=True)
    else:
        if not tuning_spec or not tuning_manifest:
            raise ValueError("tuning_spec and tuning_manifest required unless skip_tuning")

        tuning_id = f"{experiment_id}_tune"
        tuning_output = f"{output_dir}/{tuning_id}"
        print(f"\n[sequencer] === PHASE 0: TUNING ({tuning_id}) ===", flush=True)

        tuning_result = run_policy_matrix_experiment.remote(
            tuning_spec,
            tuning_manifest,
            experiment_id=tuning_id,
            start_index=start_index,
            top_k=top_k,
            output_dir=tuning_output,
            environment=environment,
        )

        learned = _extract_learned_weights_from_manifest(tuning_result)
        sequencer_manifest["phases"].append(
            {
                "phase": "tuning",
                "experiment_id": tuning_id,
                "status": "completed" if learned else "completed_no_weights",
                "learned_weights": learned,
                "result_summary": _summarize_result(tuning_result),
            }
        )

        if not learned:
            print("[sequencer][ERROR] no learned weights from tuning, aborting evals", flush=True)
            _write_sequencer_manifest(sequencer_manifest, output_dir)
            return sequencer_manifest

        _save_weights_to_volume(learned, output_dir)

        print(f"[sequencer] learned weights: {learned}", flush=True)
        print(
            f"[sequencer] cooling down {cooldown_seconds}s to let tuning containers drain...",
            flush=True,
        )
        time.sleep(cooldown_seconds)

    # ----------------------------------------------------------------
    # Phase 1..N: Evals
    # ----------------------------------------------------------------
    for eval_idx, eval_manifest in enumerate(eval_manifests):
        eval_id = f"{experiment_id}_eval{eval_idx}"
        eval_output = f"{output_dir}/{eval_id}"
        eval_label = eval_manifest.get("_note", f"eval-{eval_idx}")

        print(
            f"\n[sequencer] === PHASE {eval_idx + 1}: EVAL ({eval_id}) on {eval_label} ===",
            flush=True,
        )

        patched_eval_spec = deepcopy(eval_spec)
        for p in patched_eval_spec.get("policies", []):
            if p.get("name") == "gorgo":
                p["hyperparameters"] = learned
                print(
                    f"[sequencer] patched {p.get('label', 'gorgo')} with learned weights",
                    flush=True,
                )

        eval_result = run_policy_matrix_experiment.remote(
            patched_eval_spec,
            eval_manifest,
            experiment_id=eval_id,
            start_index=start_index,
            top_k=top_k,
            output_dir=eval_output,
            environment=environment,
        )

        sequencer_manifest["phases"].append(
            {
                "phase": f"eval{eval_idx}",
                "experiment_id": eval_id,
                "learned_weights": learned,
                "status": "completed",
                "result_summary": _summarize_result(eval_result),
            }
        )

        if eval_idx < len(eval_manifests) - 1:
            print(
                f"[sequencer] cooling down {cooldown_seconds}s before next eval...",
                flush=True,
            )
            time.sleep(cooldown_seconds)

    sequencer_manifest["completed_utc"] = _utc_now()
    _write_sequencer_manifest(sequencer_manifest, output_dir)
    print(f"\n[sequencer] all phases complete", flush=True)
    return sequencer_manifest


@app.local_entrypoint()
def sequencer(
    tuning_spec_path: str = "",
    tuning_manifest_path: str = "",
    eval_spec_path: str = "",
    eval_manifest_paths: str = "",
    experiment_id: str = "sequenced_experiment",
    output_dir: str = "/results/policy_matrix_sweep/sequenced",
    start_index: int = 0,
    top_k: int = 1,
    cooldown_seconds: float = 30.0,
    skip_tuning: bool = False,
):
    """Local entrypoint: reads specs from disk, then hands off to a
    Modal function that orchestrates all phases remotely."""

    tuning_spec = json.loads(Path(tuning_spec_path).read_text()) if tuning_spec_path else None
    tuning_manifest = (
        json.loads(Path(tuning_manifest_path).read_text()) if tuning_manifest_path else None
    )
    eval_spec = json.loads(Path(eval_spec_path).read_text()) if eval_spec_path else None
    eval_manifests = []
    if eval_manifest_paths:
        for m in eval_manifest_paths.split(","):
            m = m.strip()
            if m:
                eval_manifests.append(json.loads(Path(m).read_text()))

    ref_spec = tuning_spec or eval_spec or {}
    ref_manifest = tuning_manifest or {"top": []}
    if ref_spec:
        _validate_spec(ref_spec)
    environment = _capture_environment(ref_spec, ref_manifest)
    print(f"[sequencer] commit={environment.get('gorgo_commit', '?')}", flush=True)
    if environment.get("gorgo_dirty"):
        print("[sequencer][warn] working tree has uncommitted changes", flush=True)

    result = run_sequenced_experiment.remote(
        tuning_spec=tuning_spec,
        tuning_manifest=tuning_manifest,
        eval_spec=eval_spec,
        eval_manifests=eval_manifests,
        experiment_id=experiment_id,
        output_dir=output_dir,
        start_index=start_index,
        top_k=top_k,
        cooldown_seconds=cooldown_seconds,
        skip_tuning=skip_tuning,
        environment=environment,
    )
    print(json.dumps(result, indent=2))


def _extract_learned_weights_from_manifest(result: dict) -> dict | None:
    """Extract learned weights from the run_policy_matrix_experiment result.

    The tuning phase always writes ``learned_weights`` into the returned
    manifest dict (added to _run_policy_matrix_experiment).
    """
    if result.get("learned_weights"):
        return result["learned_weights"]

    # Fallback: check eval data
    eval_data = result.get("eval") or {}
    if eval_data.get("learned_weights"):
        return eval_data["learned_weights"]
    for ev in result.get("evals", []):
        if ev.get("learned_weights"):
            return ev["learned_weights"]

    return None


def _summarize_result(result: dict) -> dict:
    """Extract a compact summary from a run manifest."""
    timing = result.get("timing", {})
    status = result.get("status", "unknown")
    policies = []
    for p in result.get("results", {}).get("policies") or []:
        policies.append(p.get("label", "?"))
    return {
        "status": status,
        "started": timing.get("started_utc"),
        "completed": timing.get("completed_utc"),
        "policies": policies,
    }


def _write_sequencer_manifest(manifest: dict, output_dir: str) -> None:
    path = Path(output_dir) / "sequencer_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    bench_results_volume.commit()
    print(f"[sequencer] wrote {path}", flush=True)
