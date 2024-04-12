from cnnClassifier.constants import *
import os
from pathlib import Path
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareModelConfig,
                                                EvaluationConfig,
                                                TrainingConfig,
                                                PrepareCallbacksConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url= config.source_url,
            local_data_file= config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:

        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir = Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath = Path(config.checkpoint_model_filepath)
        )


        return prepare_callback_config
    
    def get_prepare_model_config(self) -> PrepareModelConfig:
            
        config = self.config.prepare_model

        create_directories([config.root_dir])
    

        prepare_model_config = PrepareModelConfig(
                root_dir = Path(config.root_dir),
                model_path = Path(config.model_path),
                updated_model_path = Path(config.updated_model_path),
                params_learning_rate = self.params.LEARNING_RATE,
                params_weights = self.params.WEIGHTS,
                params_classes = self.params.CLASSES,
                params_batch_size = self.params.BATCH_SIZE
                )
        
        return prepare_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_model
        params = self.params
        training_data = os.path.join(self.config.data_extraction.extracted_data, "training_data.csv")
        create_directories([
            Path(training.root_dir)
        ])
        target_data = os.path.join(self.config.data_extraction.extracted_data, "target_data.csv")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_model_path=Path(prepare_base_model.updated_model_path),
            training_data=Path(training_data),
            target_data=Path(target_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
        )

        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/base_model.h5",
            all_params=self.params,
            params_batch_size=self.params.EVAL_BATCH_SIZE,
            eval_data_path = "artifacts/data_extraction/extracted_data/eval_Data_X.csv",
            eval_data_target = "artifacts/data_extraction/extracted_data/eval_Data_Y.csv"
        )
        return eval_config

    

    