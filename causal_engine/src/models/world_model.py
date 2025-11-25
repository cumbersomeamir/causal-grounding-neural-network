import torch
import torch.nn as nn
from ..causal_graph.scm import SCM

class WorldModel(nn.Module):
    """
    Wrapper for the SCM that acts as the 'World Model'.
    In a more complex setup, this could include transition dynamics (P(S_t+1 | S_t, A_t)).
    For this static/tabular setup, it wraps the SCM and provides a unified interface.
    """
    def __init__(self, scm: SCM):
        super().__init__()
        self.scm = scm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scm(x)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        return self.scm.sample(n_samples)
    
    def intervene(self, intervention: dict, n_samples: int = 1) -> torch.Tensor:
        return self.scm.intervene(intervention, n_samples)

