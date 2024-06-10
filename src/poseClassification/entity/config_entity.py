from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir : Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class ConfiguredModelConfig:
    root_dir : Path
    configured_model_path: Path

@dataclass
class ModelTrainingConfig:
    root_dir : Path
    trained_model_path : Path
    configured_model_path: Path
    params_epochs: int