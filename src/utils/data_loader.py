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

    def __init__(self, config=Config()):
        self.config = config
    
    def load_raw_data(self)-> pd.DataFrame:
        df = pd.read_csv(self.config.paths.RAW_DATA_FILE, delimiter='\t', usecols=[0, 1, 2], names=['user_id', 'item_id', 'rating'])
        return df
    
    def load_train_data(self)-> pd.DataFrame:
        df = pd.read_csv(self.config.paths.TRAIN_DATA_FILE, delimiter='\t')
        return df
    
    def load_val_data(self)-> pd.DataFrame:
        df = pd.read_csv(self.config.paths.VAL_DATA_FILE, delimiter='\t')
        return df
    
    def load_numpy(self)-> tuple[np.ndarray]:
        train_df = self.load_train_data()
        val_df = self.load_val_data()
        x_train, x_test, y_train, y_test = train_df.values, val_df.values, train_df['rating'].values, val_df['rating'].values
        return x_train, x_test, y_train, y_test