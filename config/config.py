from pathlib import Path

class Config:
    """
    Configuration class for managing file paths within the project.

    Attributes:
        ROOT_PATH (Path): The root directory of the project, determined dynamically by the location of this file.
        RAW_DATA_PATH (Path): The directory path for raw data files.
        PROCESSED_DATA_PATH (Path): The directory path for processed data files.
        RAW_DATA_FILE (Path): The file path for the raw dataset file.
        TRAIN_DATA_FILE (Path): The file path for the training dataset.
        VAL_DATA_FILE (Path): The file path for the validation dataset.
        PIPELINE_PATH (Path): The directory path for storing pipeline-related files.
    """
    
    ROOT_PATH = Path(__file__).parent.parent
    RAW_DATA_PATH = ROOT_PATH / 'data' / 'raw'
    PROCESSED_DATA_PATH = ROOT_PATH / 'data' / 'processed'
    RAW_DATA_FILE = RAW_DATA_PATH / 'movielens100k.data'
    TRAIN_DATA_FILE = PROCESSED_DATA_PATH / 'train-set.csv'
    VAL_DATA_FILE = PROCESSED_DATA_PATH / 'val-set.csv'
    PIPELINE_PATH = ROOT_PATH / 'pipe'
