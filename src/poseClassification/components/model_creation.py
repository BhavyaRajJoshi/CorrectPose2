import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from poseClassification.entity.config_entity import ConfiguredModelConfig
import tensorflow
# from tensorflow.keras import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import binary_crossentropy
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import load_model



class ModelCreation:
    def __init__(self, config: ConfiguredModelConfig):
        self.config = config

    def model_creation(self):
        
        model = Sequential()
        model.add(Dense(10, activation = 'relu', input_shape=(6,)))
        model.add(Dropout(.3))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dropout(.3))
        model.add(Dense(4, activation = 'relu'))
        model.add(Dropout(.3))
        model.add(Dense(1, activation = 'sigmoid'))

        model.compile(
            optimizer = 'Adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        model.summary()
        self.save_model(path = self.config.configured_model_path, model = model)
    
    def save_model(self, path: Path, model: tensorflow.keras.Model):
        model.save(path)