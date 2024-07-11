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
        load_numpy(self): Loads training and validation data as numpy arrays and returns a tuple of four arrays.
    """
    
    @staticmethod
    def load_raw_data()-> pd.DataFrame:
        # return pd.read_csv(Config.Paths.RAW_DATA_PATH, delimiter='\t')
        pass
    
    @staticmethod
    def load_data(
        tag: Literal['movie', 'food', 'jester'],
        set_type: Literal['full', 'train', 'val']
    )-> pd.DataFrame:
        return pd.read_csv(Config.Paths.get_path(tag, set_type), delimiter='\t')

    @staticmethod
    def load_numpy(tag: Literal['movie', 'food', 'jester']) -> tuple[np.ndarray]:
        train_df = DataLoader.load_data(tag=tag, set_type='train')
        val_df = DataLoader.load_data(tag=tag, set_type='val')
        x_train, x_test, y_train, y_test = train_df.values, val_df.values, train_df['rating'].values.astype(np.float32), val_df['rating'].values.astype(np.float32)
        return x_train, y_train, x_test, y_test