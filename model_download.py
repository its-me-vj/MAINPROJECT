from huggingface_hub import snapshot_download

model_name = "rootstrap-org/crowd-counting"

# Download and save the model locally
snapshot_download(repo_id=model_name, local_dir="./crowd_counting_model")
