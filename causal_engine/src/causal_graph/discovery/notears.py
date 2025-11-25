import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any
from .base import CausalDiscoveryAlgorithm
from ...utils.logger import get_logger

logger = get_logger(__name__)

class LocallyConnectedLinear(nn.Module):
    """
    A linear layer where each output node has its own set of weights 
    connected to all input nodes, effectively simulating:
    output_j = W_j @ input
    Used to implement the adjacency masking efficiently.
    """
    def __init__(self, num_linear: int, input_features: int, output_features: int, bias: bool = True):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = np.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: [batch_size, features]
        # We want [batch_size, num_linear, output_features]
        # Reshape input: [batch_size, 1, features]
        # Weight: [num_linear, features, output_features]
        # Result: [batch_size, num_linear, output_features]
        
        # Actually, standard usage in NOTEARS for d nodes, m hidden:
        # Input is [n, d].
        # We want to process each dimension d with its own MLP.
        # But the input to the MLP for dimension j is the WHOLE vector x (masked).
        
        # Let's adapt: 
        # x: [n, d] -> [n, d, d] (repeat)
        # Linear: [d, d, m] (d independent layers, each taking d inputs)
        
        x = x.unsqueeze(1).expand(-1, self.num_linear, -1)
        out = torch.matmul(x.unsqueeze(2), self.weight).squeeze(2)
        if self.bias is not None:
            out = out + self.bias
        return out


class DeepCausalDiscovery(CausalDiscoveryAlgorithm):
    """
    Production-grade implementation of NOTEARS-MLP (Zheng et al. 2020).
    Uses Augmented Lagrangian to solve the constrained optimization problem:
    min Loss(X) s.t. h(W) = 0 (DAG constraint).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.lambda1 = config.get('lambda1', 0.01) # L1 penalty
        self.lambda2 = config.get('lambda2', 0.01) # L2 penalty
        self.w_threshold = config.get('w_threshold', 0.3)
        self.max_iter = config.get('max_iter', 100)
        self.h_tol = config.get('h_tol', 1e-8)
        self.rho_max = config.get('rho_max', 1e+16)
        
    def _init_model(self, num_nodes: int):
        # Adjacency Matrix (Learnable)
        self.adj = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.zeros_(self.adj) # Start with empty graph
        
        # Two-layer MLP for each node
        self.fc1 = LocallyConnectedLinear(num_nodes, num_nodes, self.hidden_dim)
        self.fc2 = LocallyConnectedLinear(num_nodes, self.hidden_dim, 1)
        
        self.parameters = [self.adj] + list(self.fc1.parameters()) + list(self.fc2.parameters())

    def _forward(self, x):
        # x: [n, d]
        adj = self.adj
        # fc1 takes x masked by adj.
        # We need to enforce that the j-th MLP only sees x_i if adj[i, j] != 0.
        # Our LocallyConnectedLinear takes [n, d] and multiplies by [d, d, hidden].
        # weight[j, i, k] is connection from input i to hidden k of node j.
        # We multiply weight[j] by adj[:, j].
        
        # adj is [d, d] (source, target). adj[:, j] is inputs to j.
        # We need to broadcast adj[:, j] to [d, hidden].
        
        w1 = self.fc1.weight # [d, d, hidden]
        # adj.T is [target, source] -> [d, d]
        # adj.T.unsqueeze(2) -> [d, d, 1]
        
        w1_masked = w1 * self.adj.T.unsqueeze(2)
        
        # Manual forward for fc1 with masked weights
        # x: [n, d] -> [n, d, d]
        x_expanded = x.unsqueeze(1).expand(-1, x.shape[1], -1) # [n, d_out, d_in]
        
        # [n, d_out, d_in] @ [d_out, d_in, hidden] -> [n, d_out, hidden]
        # We need batched matmul.
        # torch.bmm does [b, n, m] @ [b, m, p].
        # Here we have [d_out] as batch dimension effectively? No.
        # We have [n] as batch.
        
        # Einsum is clearer here.
        # n: batch, j: target node, i: source node, k: hidden
        h = torch.einsum('ni,jik->njk', x, w1_masked) + self.fc1.bias
        h = torch.sigmoid(h)
        
        # fc2: [n, d, hidden] -> [n, d, 1]
        # No masking needed here (internal to node j)
        # w2: [d, hidden, 1]
        x_hat = torch.einsum('njk,jkl->njl', h, self.fc2.weight) + self.fc2.bias
        x_hat = x_hat.squeeze(2) # [n, d]
        return x_hat

    def _h_func(self):
        d = self.adj.shape[0]
        M = self.adj * self.adj
        h = torch.trace(torch.matrix_exp(M)) - d
        return h

    def fit(self, data: pd.DataFrame) -> 'CausalDiscoveryAlgorithm':
        logger.info("Starting Deep Causal Discovery (NOTEARS-MLP)...")
        X = torch.tensor(data.values).float()
        n, d = X.shape
        
        self._init_model(d)
        optimizer = torch.optim.LBFGS(self.parameters, max_iter=500) # LBFGS often better for this
        
        rho, alpha, h = 1.0, 0.0, np.inf
        
        for iteration in range(self.max_iter):
            while rho < self.rho_max:
                def closure():
                    optimizer.zero_grad()
                    x_hat = self._forward(X)
                    loss_mse = 0.5 / n * torch.sum((x_hat - X) ** 2)
                    h_val = self._h_func()
                    loss_l1 = self.lambda1 * torch.sum(torch.abs(self.adj))
                    
                    # Augmented Lagrangian
                    loss = loss_mse + loss_l1 + alpha * h_val + 0.5 * rho * h_val * h_val
                    loss.backward()
                    return loss

                optimizer.step(closure)
                
                with torch.no_grad():
                    h_new = self._h_func().item()
                
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            
            h = h_new
            alpha += rho * h
            
            if h <= self.h_tol:
                break
                
            logger.info(f"Iter {iteration}: h={h:.4e}, rho={rho:.1e}")

        # Post-process adjacency matrix
        W_est = self.adj.detach().cpu().numpy()
        W_est[np.abs(W_est) < self.w_threshold] = 0
        
        self._graph = nx.DiGraph()
        labels = data.columns
        for i in range(d):
            self._graph.add_node(labels[i])
            for j in range(d):
                if W_est[i, j] != 0:
                    self._graph.add_edge(labels[i], labels[j], weight=W_est[i, j])
                    
        logger.info(f"Discovery complete. Found {self._graph.number_of_edges()} edges.")
        return self

    def get_graph(self) -> nx.DiGraph:
        if self._graph is None:
            raise ValueError("Model not fitted")
        return self._graph

