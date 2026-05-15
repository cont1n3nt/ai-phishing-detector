from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_file: Path = base_dir / "data" / "preprocess" / "cleaned_dataset.csv"
    model_dir: Path = base_dir / "model"
    images_dir: Path = base_dir / "images"
    model_filename: str = "phishing_pipeline.pkl"

    threshold: float = 0.4
    min_text_length: int = 20
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)

    hf_model_repo: str = "cont1n3nt/ai-phishing-model"
    hf_dataset_repo: str = "cont1n3nt/ai-phishing-dataset"

    model_config_path: str | None = ".env"


settings = Settings()
