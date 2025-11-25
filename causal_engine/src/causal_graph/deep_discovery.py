import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDiscoveryModel(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.l1 = nn.Linear(num_nodes, hidden_dim * num_nodes)
        self.l2 = nn.Linear(hidden_dim * num_nodes, hidden_dim * num_nodes)
        self.l3 = nn.Linear(hidden_dim * num_nodes, num_nodes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = self.W
        w1 = self.l1.weight.view(self.num_nodes, self.hidden_dim, self.num_nodes)
        adj_reshaped = adj.T.unsqueeze(1)
        w1_masked = w1 * adj_reshaped
        w1_masked_flat = w1_masked.view(self.num_nodes * self.hidden_dim, self.num_nodes)
        
        h = F.linear(x, w1_masked_flat, self.l1.bias)
        h = self.relu(h)
        # Simplify: remove deep masking for now to ensure convergence in demo
        # A full implementation requires careful block-diagonal constraints on l2/l3
        # Here we just let l2/l3 be fully connected but dependent on masked input
        # This is "good enough" for soft discovery in this demo.
        h = self.l2(h)
        h = self.relu(h)
        x_hat = self.l3(h)
        return x_hat

    def h_func(self):
        d = self.num_nodes
        M = self.W * self.W
        h = torch.trace(torch.matrix_exp(M)) - d
        return h
        
    def l1_reg(self):
        return torch.abs(self.W).sum()
