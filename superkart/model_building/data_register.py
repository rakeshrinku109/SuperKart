from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

DATASET_REPO = "rkpworks/SuperKart-Sales"

# Authenticate with Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the dataset repository exists; create it if not
try:
    api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
    print(f"Dataset repo '{DATASET_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating dataset repo '{DATASET_REPO}'...")
    create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False)
    print(f"Dataset repo '{DATASET_REPO}' created successfully.")

# Upload the raw data folder to Hugging Face
api.upload_folder(
    folder_path="superkart/data",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
