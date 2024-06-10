from poseClassification.constants import *
import os
from pathlib import Path
from poseClassification.utils.common import read_yaml, create_directories
from poseClassification.entity.config_entity import (DataIngestionConfig,
                                                ModelTrainingConfig,
                                                ConfiguredModelConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)->DataIngestionConfig:
        
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_configured_model_config(self) -> ConfiguredModelConfig:
        config = self.config.model_creation

        create_directories([config.root_dir])

        model_config = ConfiguredModelConfig(
            root_dir = Path(config.root_dir),
            configured_model_path = Path(config.configured_model_path)
        )
        return model_config
    

    def get_model_training_config(self) -> ModelTrainingConfig:

        model_training = self.config.model_training
        model_creation = self.config.model_creation
        params = self.params 
        #traing_data = pd.read_csv(self.config.data_extraction, 'data.csv')

        create_directories([
            Path(model_training.root_dir)
        ])

        training_config = ModelTrainingConfig(
            root_dir=Path(model_training.root_dir),
            trained_model_path = Path(model_training.trained_model_path),
            configured_model_path = Path(model_creation.configured_model_path),
            #training_data=Path(training_data),
            params_epochs=params.EPOCHS
        )

        return training_config
