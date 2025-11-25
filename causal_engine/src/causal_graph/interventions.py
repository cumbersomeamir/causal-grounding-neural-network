from .scm import SCM
from typing import Dict
import torch
import pandas as pd

def do_intervention(scm: SCM, intervention: Dict[str, float], n_samples: int = 100) -> pd.DataFrame:
    """
    Wrapper for SCM intervention logic returning a DataFrame.
    """
    samples = scm.intervene(intervention, n_samples)
    df = pd.DataFrame(samples.numpy(), columns=scm.nodes)
    return df

def get_counterfactual(scm: SCM, observation_df: pd.DataFrame, intervention: Dict[str, float]) -> pd.DataFrame:
    """
    Computes counterfactual for a specific observation.
    """
    obs_tensor = torch.tensor(observation_df.iloc[0].values).float().unsqueeze(0)
    cf_tensor = scm.counterfactual(obs_tensor, intervention)
    return pd.DataFrame(cf_tensor.numpy(), columns=scm.nodes)

