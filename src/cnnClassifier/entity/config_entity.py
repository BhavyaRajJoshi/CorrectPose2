from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_url : str
    local_data_file: Path
    unzip_dir : Path



@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    model_path: Path
    updated_model_path: Path
    params_learning_rate: float
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int


@dataclass(frozen = True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path