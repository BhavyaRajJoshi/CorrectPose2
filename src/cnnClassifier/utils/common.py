import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
#from boxsdk import ConfigBox


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """READS YAML file AND returns
    
    Args:
        'path_to_yaml':(string): path like input
    
    Raises:
        ValueError : ifyaml file is empty
        e: empty file
    Returns:
        config_box: config_box type
    """

    try: 
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    ''' create list of directories
    
    Args:
        path_to_directories(list): list of path of directories to be created
        ignore_log(bool, optional): ignore if multiple dirs is to be created. default to False'''

    for path in path_to_directories:
        os.makedirs(path, exist_ok = True)
        if verbose:
            logger.info(f"creating directory at path: {path}")
    


@ensure_annotations
def save_json(path : Path, data: dict):
    #while doing model evaluation
    #when we create a matrix we will be saving the loss and accuracy in 
    #json format
    '''save json data
    
    Args:
        path(Path): path to json file
        data(dict): data to be saved in json file
    '''       
    with open(path, 'w') as f:
        json.dump(data, f , indent = 4 )

    logger.info(f"json saved at : {path}")


@ensure_annotations
def  load_json(path: Path) -> ConfigBox:
    '''
    load json file data
    
    args :
        path(Path): path to json file
        
    returns:
        config_box: data as class attribute instead of dictionary
    '''

    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded from : {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    joblib.dump(value = data, filename = path)
    logger.info(f"bin file saved at : {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:

    '''Load bin data
    
    args:
        path(Path): path to binary file
        
    return:
        any: object stored in file 
        
    '''

    data = joblib.load(path)
    logger.info(f"binary file loaded from : {path}")
    return data

@ensure_annotations
def get_size( path : Path) -> str:

    size_in_kb = round(os.path.getsize(path)/1024)
    return f"{size_in_kb} KB"


def decode_image(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encode_imageIntoBase64(croppedImagePath):
    with open(croppedImagePath, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string