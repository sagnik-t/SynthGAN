from dotenv import set_key
from typing import Literal

from utils.data_loader import DataLoader

def fetch_metadata(data: Literal['train', 'val']) -> tuple[int]:
    """
    Fetches metadata from the train or val data file.
    
    Args:
    data: Literal['train', 'val'], type of dataset
    
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
    
    # Update or create .env file with environment variables
    set_key('.env', 'TRAIN_NUM_USERS', str(train_vars[0]))
    set_key('.env', 'TRAIN_NUM_ITEMS', str(train_vars[1]))
    set_key('.env', 'VAL_NUM_USERS', str(val_vars[0]))
    set_key('.env', 'VAL_NUM_ITEMS', str(val_vars[1]))
    
    return

if __name__ == '__main__':
    set_env_vars()
