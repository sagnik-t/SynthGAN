from pathlib import Path

class Config:
    
    ROOT_PATH = Path(__file__).parent.parent
    RAW_DATA_PATH = ROOT_PATH / 'data' / 'raw'
    PROCESSED_DATA_PATH = ROOT_PATH / 'data' / 'processed'
    RAW_DATA_FILE = RAW_DATA_PATH / 'movielens100k.data'
    TRAIN_DATA_FILE = PROCESSED_DATA_PATH / 'train-set.csv'
    VAL_DATA_FILE = PROCESSED_DATA_PATH / 'val-set.csv'
    PIPELINE_PATH = ROOT_PATH / 'pipe'
