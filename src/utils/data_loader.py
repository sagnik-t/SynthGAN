import pandas as pd
from pathlib import Path

from config import Config

class DataLoader:
    
    def __init__(self, config=Config):
        self.config = config
    
    def load_raw_data(self):
        df = pd.read_csv(self.config.RAW_DATA_FILE, delimiter='\t', usecols=[0, 1, 2], names=['user_id', 'item_id', 'rating'])
        return df
    
    def load_train_data(self):
        df = pd.read_csv(self.config.TRAIN_DATA_FILE, delimiter='\t')
        return df
    
    def load_val_data(self):
        df = pd.read_csv(self.config.VAL_DATA_FILE, delimiter='\t')
        return df
