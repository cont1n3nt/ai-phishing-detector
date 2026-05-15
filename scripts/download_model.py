"""Download trained model from HuggingFace Hub."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from config import settings

REPO_ID = settings.hf_model_repo
MODEL_FILENAME = settings.model_filename


def download_model():
    """Download the trained model from HuggingFace."""
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=settings.model_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to: {path}")


if __name__ == "__main__":
    download_model()
