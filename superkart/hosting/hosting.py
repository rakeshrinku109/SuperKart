from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Push all deployment files to the Hugging Face Space
api.upload_folder(
    folder_path="superkart/deployment",
    repo_id="rkpworks/SuperKart-Sales",
    repo_type="space",
    path_in_repo="",
)
