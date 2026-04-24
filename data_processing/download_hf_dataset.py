"""Download a Hugging Face dataset onto the shared ``hf_datasets`` Modal volume.

Pass a hub id (``org/name``) or a dataset page URL. Data is stored under
``/data/datasets/<org>__<name>/`` so you can keep several datasets on one volume.

Examples::

    modal run data_processing/download_hf_dataset.py::download_hf_cli
    modal run --env=GORGO data_processing/download_hf_dataset.py::download_hf_cli --dataset allenai/WildChat-4.8M

Large downloads (multi-million rows): use ``--detach`` so the job keeps running if
your laptop sleeps or the CLI disconnects::

    modal run --detach --env=GORGO data_processing/download_hf_dataset.py::download_hf_cli --dataset allenai/WildChat-4.8M

Gated datasets: accept the license on the Hub and create a Modal secret (e.g.
``modal secret create --env=GORGO huggingface HF_TOKEN=hf_...``) named
``huggingface`` with ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``.
"""

from __future__ import annotations

import os
import re

import modal

from app import app, hf_datasets_volume

image = (
    modal.Image.debian_slim()
    .pip_install("datasets>=3.0", "pyarrow", "huggingface_hub")
    .add_local_python_source("app")
)

DATA_ROOT = "/data/datasets"


def parse_dataset_ref(ref: str) -> str:
    """Return ``org/name`` from a hub id or ``https://huggingface.co/datasets/...`` URL."""
    s = ref.strip()
    if not s:
        raise ValueError("dataset ref is empty")
    if s.startswith("http"):
        # https://huggingface.co/datasets/org/name[/tree/...]
        m = re.search(r"/datasets/([^/]+)/([^/?#]+)", s)
        if not m:
            raise ValueError(f"could not parse dataset id from URL: {ref!r}")
        return f"{m.group(1)}/{m.group(2)}"
    if "/" not in s:
        raise ValueError(f"expected org/name or huggingface.co/datasets URL, got {ref!r}")
    return s


def dataset_slug(dataset_id: str) -> str:
    """Filesystem-safe directory name for a hub id (one dir per dataset)."""
    return dataset_id.replace("/", "__")


def out_dir_for(dataset_id: str) -> str:
    return os.path.join(DATA_ROOT, dataset_slug(dataset_id))


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _already_saved(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "dataset_dict.json"))


@app.function(
    image=image,
    volumes={"/data": hf_datasets_volume},
    timeout=86400,
    memory=64 * 1024,
    retries=5,
    secrets=[modal.Secret.from_name("hugging_face")],
)
def download_hf_dataset(
    dataset: str,
    *,
    force: bool = False,
    config: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    import shutil
    import time

    from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset

    dataset_id = parse_dataset_ref(dataset)
    out = out_dir_for(dataset_id)

    os.environ.setdefault("HF_HOME", "/data/.hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data/.hf_datasets_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/.hf_hub_cache")

    if _already_saved(out) and not force:
        print(f"Dataset already at {out} (pass force=True to replace).")
        return {"dataset_id": dataset_id, "path": out, "rows": 0, "skipped": True}

    if force and os.path.isdir(out):
        shutil.rmtree(out)

    token = _hf_token()
    if not token:
        print(
            "HF_TOKEN not set: if download fails with 401/403, accept the license on the Hub "
            f"and set HF_TOKEN (dataset page: https://huggingface.co/datasets/{dataset_id})."
        )

    os.makedirs(DATA_ROOT, exist_ok=True)

    t0 = time.time()
    dc = DownloadConfig(resume_download=True)
    kw: dict = {
        "token": token,
        "download_config": dc,
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        kw["revision"] = revision

    if config is not None:
        ds = load_dataset(dataset_id, config, **kw)
    else:
        ds = load_dataset(dataset_id, **kw)

    ds.save_to_disk(out)
    hf_datasets_volume.commit()
    elapsed = time.time() - t0

    if isinstance(ds, DatasetDict):
        n = sum(len(ds[k]) for k in ds)
    elif isinstance(ds, Dataset):
        n = len(ds)
    else:
        n = -1
    print(f"Saved {dataset_id} to {out} ({n} rows total) in {elapsed:.1f}s")
    return {"dataset_id": dataset_id, "path": out, "rows": n, "skipped": False}


@app.local_entrypoint()
def download_hf_cli(
    dataset: str = "lmsys/lmsys-chat-1m",
    force: bool = False,
    config: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    download_hf_dataset.remote(
        dataset,
        force=force,
        config=config,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
