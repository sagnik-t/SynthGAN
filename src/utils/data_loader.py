import pandas as pd

from config import Config

class DataLoader:
    """
    A class for loading raw, train, and validation data.

    Attributes:
        config (Config): An instance of the Config class containing the paths to the data files.

    Methods:
        __init__(self, config=Config): Initializes the DataLoader with a Config instance.
        load_raw_data(self): Loads raw data from a specified file and returns a pandas DataFrame.
        load_train_data(self): Loads training data from a specified file and returns a pandas DataFrame.
        load_val_data(self): Loads validation data from a specified file and returns a pandas DataFrame.
    """

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
