import torch
import torch.optim as optim
import numpy as np
from .deep_discovery import CausalDiscoveryModel
from ..utils.logger import get_logger

logger = get_logger(__name__)

def train_deep_discovery(data: torch.Tensor, epochs: int = 100, lambda1: float = 0.1, lambda2: float = 0.1):
    """
    Trains the continuous structure learning model.
    """
    num_nodes = data.shape[1]
    model = CausalDiscoveryModel(num_nodes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Lagrangian dual ascent variables
    alpha = 0.0
    rho = 1.0
    h_tol = 1e-8
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        x_hat = model(data)
        
        # Reconstruction loss (MSE)
        loss_recon = 0.5 / data.shape[0] * ((x_hat - data) ** 2).sum()
        
        # DAG constraint
        h_val = model.h_func()
        
        # L1 regularization on W to encourage sparsity
        loss_l1 = model.l1_reg()
        
        # Augmented Lagrangian
        loss = loss_recon + lambda1 * loss_l1 + alpha * h_val + 0.5 * rho * h_val * h_val
        
        loss.backward()
        optimizer.step()
        
        # Update rho and alpha (dual ascent steps) simplified
        if epoch % 10 == 0:
             with torch.no_grad():
                h_new = model.h_func().item()
                if h_new > 0.25 * h_val.item():
                    rho *= 10
                else:
                    break # Or standard augmented lagrangian update
                alpha += rho * h_new

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}, h={h_val.item():.4f}, W_l1={loss_l1.item():.4f}")
            
    return model.W.detach()

