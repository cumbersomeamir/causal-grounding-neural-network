from .scm import SCM
import pandas as pd
import torch

def ancestral_sampling(scm: SCM, n_samples: int = 100) -> pd.DataFrame:
    """
    Performs ancestral sampling from the SCM.
    """
    samples = scm.sample(n_samples)
    return pd.DataFrame(samples.numpy(), columns=scm.nodes)

