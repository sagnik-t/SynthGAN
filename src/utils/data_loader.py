from typing import Literal
import numpy as np
import pandas as pd

from config import Config

class DataLoader:
    """
    A class for loading raw, train, and validation data.

    Attributes:
        config (Config): An instance of the Config class containing the paths to the data files.

    Methods:
        load_raw_data(self): Loads raw data from a specified file and returns a pandas DataFrame.
        load_train_data(self): Loads training data from a specified file and returns a pandas DataFrame.
        load_val_data(self): Loads validation data from a specified file and returns a pandas DataFrame.
        load_numpy(self): Loads training and validation data as numpy arrays and returns a tuple of four arrays.
    """
    
    @staticmethod
    def load_raw_data()-> pd.DataFrame:
        return pd.read_csv(Config.Paths.RAW_DATA_PATH, delimiter='\t')
    
    @staticmethod
    def load_data(set_type: Literal['full', 'train', 'val'])-> pd.DataFrame:
        return pd.read_csv(Config.Paths.DATA_PATH[f'{set_type}-set'])