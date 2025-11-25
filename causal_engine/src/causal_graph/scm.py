import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class RootNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(1))
        
    def forward(self, x=None):
        return self.mean

class SCM(nn.Module):
    def __init__(self, graph: nx.DiGraph, hidden_dim: int = 64):
        super().__init__()
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.models = nn.ModuleDict()
        self.sorted_nodes = list(nx.topological_sort(graph))
        
        for node in self.nodes:
            parents = list(graph.predecessors(node))
            input_dim = len(parents)
            if input_dim > 0:
                self.models[node] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            else:
                # Root node
                self.models[node] = RootNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicts values given parents in the provided x (teacher forcing).
        """
        predictions = {}
        for node in self.nodes:
            parents = list(self.graph.predecessors(node))
            if len(parents) > 0:
                parent_indices = [self.node_to_idx[p] for p in parents]
                parent_vals = x[:, parent_indices]
                predictions[node] = self.models[node](parent_vals)
            else:
                # Expand parameter to batch size
                predictions[node] = self.models[node]().expand(x.shape[0], 1)
        
        # Stack predictions in correct order
        output = torch.cat([predictions[n] for n in self.nodes], dim=1)
        return output

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Ancestral sampling.
        """
        samples = torch.zeros((n_samples, len(self.nodes)))
        sample_dict = {}
        
        with torch.no_grad():
            for node in self.sorted_nodes:
                parents = list(self.graph.predecessors(node))
                if len(parents) > 0:
                    parent_vals = torch.cat([sample_dict[p] for p in parents], dim=1)
                    mean = self.models[node](parent_vals)
                    noise = torch.randn_like(mean) * 0.1
                    sample_dict[node] = mean + noise
                else:
                    mean = self.models[node]().expand(n_samples, 1)
                    noise = torch.randn_like(mean) * 0.1
                    sample_dict[node] = mean + noise
                    
                idx = self.node_to_idx[node]
                samples[:, idx] = sample_dict[node].squeeze()
                
        return samples

    def intervene(self, intervention_dict: Dict[str, float], n_samples: int = 1) -> torch.Tensor:
        """
        Perform hard intervention do(X=x).
        """
        samples = torch.zeros((n_samples, len(self.nodes)))
        sample_dict = {}
        
        with torch.no_grad():
            for node in self.sorted_nodes:
                if node in intervention_dict:
                    val = torch.tensor(intervention_dict[node]).float().expand(n_samples, 1)
                    sample_dict[node] = val
                else:
                    parents = list(self.graph.predecessors(node))
                    if len(parents) > 0:
                        parent_vals = torch.cat([sample_dict[p] for p in parents], dim=1)
                        mean = self.models[node](parent_vals)
                        noise = torch.randn_like(mean) * 0.1
                        sample_dict[node] = mean + noise
                    else:
                        mean = self.models[node]().expand(n_samples, 1)
                        noise = torch.randn_like(mean) * 0.1
                        sample_dict[node] = mean + noise
                
                idx = self.node_to_idx[node]
                samples[:, idx] = sample_dict[node].squeeze()
                
        return samples

    def counterfactual(self, observation: torch.Tensor, intervention_dict: Dict[str, float]) -> torch.Tensor:
        # 1. Abduction
        noises = {}
        obs_dict = {node: observation[0, i].view(1, 1) for i, node in enumerate(self.nodes)}
        
        with torch.no_grad():
            for node in self.sorted_nodes:
                parents = list(self.graph.predecessors(node))
                if len(parents) > 0:
                    parent_vals = torch.cat([obs_dict[p] for p in parents], dim=1)
                    pred_mean = self.models[node](parent_vals)
                    noises[node] = obs_dict[node] - pred_mean
                else:
                    pred_mean = self.models[node]().expand(1, 1)
                    noises[node] = obs_dict[node] - pred_mean

        # 2. & 3. Action & Prediction
        cf_samples = torch.zeros((1, len(self.nodes)))
        cf_dict = {}
        
        with torch.no_grad():
            for node in self.sorted_nodes:
                if node in intervention_dict:
                    val = torch.tensor(intervention_dict[node]).float().view(1, 1)
                    cf_dict[node] = val
                else:
                    parents = list(self.graph.predecessors(node))
                    if len(parents) > 0:
                        parent_vals = torch.cat([cf_dict[p] for p in parents], dim=1)
                        mean = self.models[node](parent_vals)
                        cf_dict[node] = mean + noises[node]
                    else:
                        mean = self.models[node]().expand(1, 1)
                        cf_dict[node] = mean + noises[node]
                
                idx = self.node_to_idx[node]
                cf_samples[:, idx] = cf_dict[node].squeeze()
        
        return cf_samples
