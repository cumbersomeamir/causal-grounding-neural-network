import pytest
import torch
from src.models.causal_vae import CausalVAE

def test_causal_vae_shape():
    input_dim = 10
    model = CausalVAE(input_dim=input_dim)
    x = torch.randn(5, input_dim)
    
    recon_x, mu, logvar = model(x)
    assert recon_x.shape == (5, input_dim)
    assert mu.shape == (5, model.latent_dim)

