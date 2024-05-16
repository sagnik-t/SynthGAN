from dataclasses import dataclass, fields
import os
from pathlib import Path

class Config:
    
    @dataclass
    class Paths:
        ROOT_PATH: Path = Path(__file__).parent.parent
        RAW_DATA_PATH: Path = ROOT_PATH / 'data' / 'raw'
        PROCESSED_DATA_PATH: Path = ROOT_PATH / 'data' / 'processed'
        RAW_DATA_FILE: Path = RAW_DATA_PATH / 'movielens100k.data'
        TRAIN_DATA_FILE: Path = PROCESSED_DATA_PATH / 'train-set.csv'
        VAL_DATA_FILE: Path = PROCESSED_DATA_PATH / 'val-set.csv'
        PIPELINE_PATH: Path = ROOT_PATH / 'pipe'
    
    @dataclass
    class Vars:
        TRAIN_NUM_USERS: int = int(os.getenv('TRAIN_NUM_USERS'))
        TRAIN_NUM_ITEMS: int = int(os.getenv('TRAIN_NUM_ITEMS'))
        VAL_NUM_USERS: int = int(os.getenv('VAL_NUM_USERS'))
        VAL_NUM_ITERMS: int = int(os.getenv('VAL_NUM_ITEMS'))
    
    def __init__(self):
        self.paths = self.Paths()
        self.vars = self.Vars()

if __name__=='__main__':
    config = Config()
    for path in fields(config.paths):
        print(getattr(config.paths, path.name))
    for var in fields(config.vars):
        print(getattr(config.vars, var.name))