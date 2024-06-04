from dataclasses import dataclass, fields
import os
from pathlib import Path

class Config:
    """
    A configuration class that encapsulates two nested dataclasses: Paths and Vars.
    Paths contains configurations related to file and directory paths within the project.
    Vars contains configurations for various operational parameters, typically sourced from environment variables.
    
    Attributes:
        paths (Paths): An instance of the nested Paths dataclass containing path-related configurations.
        vars (Vars): An instance of the nested Vars dataclass containing operational parameters.
    """
    
    @dataclass(frozen=True)
    class Paths:
        """
        A dataclass for managing file and directory paths in the project.
        
        Attributes:
            ROOT_PATH (Path): The root directory of the project.
            RAW_DATA_PATH (Path): The directory path for raw data files.
            PROCESSED_DATA_PATH (Path): The directory path for processed data files.
            RAW_DATA_FILE (Path): The file path for the raw data file.
            TRAIN_DATA_FILE (Path): The file path for the training data set.
            VAL_DATA_FILE (Path): The file path for the validation data set.
            PIPELINE_PATH (Path): The directory path for the data processing pipeline.
        """
        ROOT_PATH: Path = Path(__file__).parent.parent
        RAW_DATA_PATH: Path = ROOT_PATH / 'data' / 'raw'
        PROCESSED_DATA_PATH: Path = ROOT_PATH / 'data' / 'processed'
        RAW_DATA_FILE: Path = RAW_DATA_PATH / 'movielens100k.data'
        TRAIN_DATA_FILE: Path = PROCESSED_DATA_PATH / 'train-set.csv'
        VAL_DATA_FILE: Path = PROCESSED_DATA_PATH / 'val-set.csv'
    
    
    @dataclass(frozen=True)
    class Vars:
        """
        A dataclass for managing operational parameters, typically sourced from environment variables.
        
        Attributes:
            NUM_USERS (int): The number of users in the dataset.
            NUM_ITEMS (int): The number of items in the dataset.
        """
        NUM_USERS: int = int(os.getenv('NUM_USERS'))
        NUM_ITEMS: int = int(os.getenv('NUM_ITEMS'))
    
    
    def __init__(self):
        self.paths = self.Paths()
        self.vars = self.Vars()


if __name__=='__main__':
    config = Config()
    for path in fields(config.paths):
        print(getattr(config.paths, path.name))
    for var in fields(config.vars):
        print(getattr(config.vars, var.name))