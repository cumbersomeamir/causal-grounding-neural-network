import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.json'):
            self.data = pd.read_json(self.file_path)
        else:
            raise ValueError("Unsupported file format")
        return self.data

    def get_data(self) -> pd.DataFrame:
        if self.data is None:
            return self.load_data()
        return self.data

def split_data(data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(data, test_size=test_size, random_state=seed)
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, random_state=seed)
    return train, val, test

