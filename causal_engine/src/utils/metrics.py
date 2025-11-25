import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def compute_kl_divergence(p, q):
    # Assumes p and q are probabilities or log-probs depending on usage, 
    # simplified here for numpy arrays
    return np.sum(np.where(p != 0, p * np.log(p / (q + 1e-10)), 0))

