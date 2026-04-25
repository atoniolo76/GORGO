# Modal quickstart for GORGO

How to run Modal jobs against the right workspace and environment for this
project, without stepping on other Claude agents that share this box.

## Project Modal coordinates

| | |
|---|---|
| Workspace | `arcadia-research` |
| Environment | `GORGO` |
| Web UI | <https://modal.com/apps/arcadia-research/GORGO> |
| HF token (in Modal secrets) | `hf_token_rome` exposes `HF_TOKEN_ROME` |
| GitHub token (in Modal secrets) | `gh_token_rome` exposes `GH_TOKEN_ROME` |

## Golden-rule invocation

Every Modal command in this project should look like this:

```bash
MODAL_PROFILE=arcadia-research modal run --env=GORGO scripts/<your_script>.py
```

- `MODAL_PROFILE=arcadia-research` selects the workspace **per-process**. It
  does not touch `~/.modal.toml`. Other agents on the same box stay on their
  own profiles.
- `--env=GORGO` selects the environment **per-invocation**. Every Modal
  subcommand (`run`, `volume ls`, `logs`, `app list`, `secret list`, ...)
  takes `--env`.

**Never run `modal profile activate <X>`** on a shared workstation. It writes
`active = true` into `~/.modal.toml` globally, and every other agent's next
`modal` call silently inherits the change. This has bitten this project
before — see `feedback_modal_shared_workstation.md` in scout's memory.

## In-script defaults

When you write a Modal app, hard-code the env so the script doesn't depend on
the caller passing `--env`:

```python
import modal

app = modal.App("gorgo-<name>", environment_name="GORGO")
```

That makes the script self-documenting and immune to whichever profile is
"active" globally.

## Secrets pattern

Both HF and GH tokens are project-scoped Modal secrets. Inside a function:

```python
@app.function(
    secrets=[
        modal.Secret.from_name("hf_token_rome"),  # provides HF_TOKEN_ROME
    ],
    gpu="A100-80GB",
    timeout=4 * 3600,
)
def my_func():
    import os
    # vLLM / huggingface_hub expects HF_TOKEN, not HF_TOKEN_ROME
    os.environ["HF_TOKEN"] = os.environ["HF_TOKEN_ROME"]
    ...
```

The remap `HF_TOKEN_ROME → HF_TOKEN` is required because that's what
`huggingface_hub`, `vllm`, and `datasets` look for. Do it inside the function
body, before any HF library import that might cache the value.

The same pattern applies to `gh_token_rome` if you ever need to push from
inside a container — though by default polecats push from the laptop, so
GH-in-container is unusual.

## Common operations

```bash
# List apps in the GORGO env
MODAL_PROFILE=arcadia-research modal app list --env=GORGO

# Tail logs of a running app
MODAL_PROFILE=arcadia-research modal logs <app-id> --env=GORGO

# List volumes
MODAL_PROFILE=arcadia-research modal volume list --env=GORGO

# Pull a file off a volume
MODAL_PROFILE=arcadia-research modal volume get <vol-name> /path/to/file --env=GORGO

# Verify your secret is wired up
MODAL_PROFILE=arcadia-research modal secret list --env=GORGO
```

## Cost discipline

- Always set `timeout=` on `@app.function()` — the project default is
  4 hours (`14400`). This is a hard ceiling that prevents runaway cost.
- Single-seed sweeps target ~$11 / Llama-3.1-8B / A100-80GB. Three-seed
  replication is ~3× and **requires re-approval** per the bead workflow.
- `modal app stop <app-id>` on the way out if you abandoned a long-running
  container.

## Known gates (HF license acceptance)

Several models / datasets used here are gated on Hugging Face. The HF account
behind `hf_token_rome` must accept each license **manually in the browser**
before that resource is fetchable from inside Modal containers. Symptoms:
`403 Forbidden` on config fetch, `DatasetNotFoundError: Dataset is a gated
dataset...`.

Currently approved (as of 2026-04-25):
- `lmsys/lmsys-chat-1m`
- (pending verification) `meta-llama/Llama-3.1-8B-Instruct`

If you hit a 403 / gated error inside a Modal container, the fix is on the
HF account, not on Modal. Visit the resource page in a browser, click
**Request access**, wait for approval (Meta is usually minutes; some are
manual review).

## Existing scripts in this repo

| Script | What it does |
|---|---|
| `scripts/fetch_lmsys_modal.py` | CPU function: fetches `lmsys/lmsys-chat-1m`, runs `dataset_metrics.py`, returns artifacts to host. |
| `scripts/calibrate.py` | A100-80GB function: deploys vLLM + Llama-3.1-8B, runs the prefill / decode@b=1 / decode-batch sweeps, writes JSONL to a Modal Volume, mirrors back to host, fits the 5 cost-model coefficients, emits `configs/calibrated_a100.yaml` + `fit_summary.json`. Includes a separate `fetch` entrypoint for resuming a sweep that disconnected mid-run. |

## Approval gate for spend

Per the project workflow, any Modal invocation that consumes meaningful
GPU time (anything beyond image builds + a 403 attempt) needs explicit scout
or human approval before running. The bead system carries this — Modal-spend
beads are tagged `human` and unblocked only after the cost is signed off.
