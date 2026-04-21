import modal

ENVIRONMENT_NAME = "alessio-dev"

app = modal.App(name="GORGO")
replicas = modal.Dict.from_name(
    "GORGO-replicas", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
completions_volume = modal.Volume.from_name(
    "GORGO-glm5-completions", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
hf_datasets_volume = modal.Volume.from_name(
    "GORGO-hf-datasets", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
