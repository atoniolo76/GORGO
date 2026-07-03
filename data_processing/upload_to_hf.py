"""Upload the packaged week release from the Modal volume to the HuggingFace Hub.

Pulls ``/data/mooncake_traces/week_release`` (per-day parquet + Mooncake JSONL)
from the ``GORGO-completions`` volume and uploads it to a dataset repo,
authenticating with the ``huggingface-secret`` Modal secret.

Usage::

    # full folder (parquet + jsonl):
    modal run --env=alessio-dev data_processing/upload_to_hf.py::main

    # parquet only (smaller, canonical HF format):
    modal run --env=alessio-dev data_processing/upload_to_hf.py::main --only-parquet
"""

from __future__ import annotations

import json

import modal

from app import app, completions_volume

image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub>=0.26", "hf_transfer")
    # Disable the Xet storage backend (its write-token endpoint 403s with a
    # standard write token); fall back to regular LFS multipart upload.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("app")
)

RELEASE_DIR = "/data/mooncake_traces/week_release"
DEFAULT_REPO = "alessiotoniolo/ART-Chat-2.5M"


def _token() -> str:
    import os

    for k in (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_API_TOKEN",
    ):
        v = os.environ.get(k)
        if v:
            return v
    present = [k for k in os.environ if "HF" in k.upper() or "HUGG" in k.upper()]
    raise RuntimeError(f"No HF token env var found. HF-ish env vars present: {present}")


@app.function(
    image=image,
    timeout=24 * 3600,
    volumes={"/data": completions_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def upload(
    folder_path: str = RELEASE_DIR,
    repo_id: str = DEFAULT_REPO,
    only_parquet: bool = False,
) -> dict:
    import os

    from huggingface_hub import HfApi

    token = _token()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    if only_parquet:
        sub = os.path.join(folder_path, "parquet")
        api.upload_folder(
            folder_path=sub,
            path_in_repo="parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add per-day parquet shards (full week, request-row trace)",
        )
    else:
        # Resumable, multi-threaded; appropriate for the ~150GB JSONL + parquet.
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            repo_type="dataset",
            print_report=True,
        )

    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "only_parquet": only_parquet,
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def whoami() -> dict:
    """Diagnostic: report the token's identity and permission scope."""
    from huggingface_hub import HfApi

    token = _token()
    info = HfApi(token=token).whoami()
    auth = info.get("auth", {})
    return {
        "name": info.get("name"),
        "type": info.get("type"),
        "token_role": auth.get("accessToken", {}).get("role"),
        "fineGrained": auth.get("accessToken", {}).get("fineGrained"),
        "orgs": [o.get("name") for o in info.get("orgs", [])],
    }


@app.local_entrypoint()
def check():
    print(json.dumps(whoami.remote(), indent=2))


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret-2")])
def list_repo(repo_id: str = DEFAULT_REPO) -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    files = api.list_repo_files(repo_id, repo_type="dataset")
    return {"repo_id": repo_id, "n_files": len(files), "files": sorted(files)}


@app.local_entrypoint()
def verify(repo_id: str = DEFAULT_REPO):
    print(json.dumps(list_repo.remote(repo_id=repo_id), indent=2))


@app.local_entrypoint()
def main(
    folder_path: str = RELEASE_DIR,
    repo_id: str = DEFAULT_REPO,
    only_parquet: bool = False,
):
    print(
        json.dumps(
            upload.remote(folder_path=folder_path, repo_id=repo_id, only_parquet=only_parquet),
            indent=2,
        )
    )
