import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.fitted = False
        self.columns = None

    def fit(self, data: pd.DataFrame):
        self.columns = data.columns
        self.imputer.fit(data)
        self.scaler.fit(data)
        self.fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        data_imputed = self.imputer.transform(data)
        data_scaled = self.scaler.transform(data_imputed)
        return pd.DataFrame(data_scaled, columns=self.columns)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

