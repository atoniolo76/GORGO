import modal

ENVIRONMENT_NAME = "alessio-dev"

app = modal.App(name="GORGO")
replicas = modal.Dict.from_name(
    "GORGO-replicas", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
completions_volume = modal.Volume.from_name(
    "GORGO-glm5-completions", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
# Output destination for ``proxy/workload.py`` runs (one JSON doc per run
# under ``/results``).
bench_results_volume = modal.Volume.from_name(
    "GORGO-bench-results", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
hf_datasets_volume = modal.Volume.from_name(
    "GORGO-hf-datasets", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
# HF ``save_to_disk`` for lmsys/lmsys-chat-1m (e.g. ``…/lmsys-chat-1m/train/*.arrow``).
lmsys_chat_1m_volume = modal.Volume.from_name(
    "GORGO-lmsys-chat-1m", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
