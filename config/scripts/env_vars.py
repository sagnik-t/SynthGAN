import os
import pandas as pd
from pathlib import Path
from typing import Literal

from config import Config
from utils.data_loader import DataLoader

def fetch_metadata(data: Literal['train'] | Literal['val']) -> tuple[int]:
    """
    Fetches metadata from the train or val data file.
    
    Args:
    data: str, type of dataset
    
    Returns:
    tuple, metadata from the data file.
    
    Raises:
    ValueError: If data is neither 'train' nor 'val'.
    """
    data_loader = DataLoader()
    
    if data == 'train':
        df = data_loader.load_train_data()
    elif data == 'val':
        df = data_loader.load_val_data()
    else:
        raise ValueError('Dataset must be either train or validation')
    
    num_users, num_items = df['user_id'].nunique(), df['item_id'].nunique()
    
    return num_users, num_items

def set_env_vars():
    """
    Sets the environment variables for train and val data.
    """
    
    train_vars = fetch_metadata('train')
    val_vars = fetch_metadata('val')
    
    os.environ['TRAIN_NUM_USERS'] = str(train_vars[0])
    os.environ['TRAIN_NUM_ITEMS'] = str(train_vars[1])
    os.environ['VAL_NUM_USERS'] = str(val_vars[0])
    os.environ['VAL_NUM_ITEMS'] = str(val_vars[1])
    
    return

if __name__ == '__main__':
    set_env_vars()