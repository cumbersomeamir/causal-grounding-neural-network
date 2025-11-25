import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

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
            # Model models P(Node | Parents)
            # For a continuous variable, we predict the mean (and optionally variance, but simplified to mean here)
            # If input_dim is 0, it's a root node, modeled as a learnable parameter or distribution
            if input_dim > 0:
                self.models[node] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            else:
                # Root node: simplified as a learnable bias for mean (assuming normalized data ~ N(0,1) initially)
                # In a real generative model, this would be a sampler.
                # Here we implement it as a simple parameter for the mean
                self.models[node] = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass essentially computes the likelihood or error.
        But SCMs usually generate data.
        This method predicts values given parents in the provided x (teacher forcing).
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
                predictions[node] = self.models[node].expand(x.shape[0], 1)
        
        # Stack predictions in correct order
        output = torch.cat([predictions[n] for n in self.nodes], dim=1)
        return output

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Ancestral sampling.
        """
        samples = torch.zeros((n_samples, len(self.nodes)))
        
        # We need to store values by node name to lookup parents easily
        sample_dict = {}
        
        with torch.no_grad():
            for node in self.sorted_nodes:
                parents = list(self.graph.predecessors(node))
                if len(parents) > 0:
                    parent_vals = torch.cat([sample_dict[p] for p in parents], dim=1)
                    # Add noise for probabilistic nature
                    mean = self.models[node](parent_vals)
                    noise = torch.randn_like(mean) * 0.1 # Assuming small noise variance
                    sample_dict[node] = mean + noise
                else:
                    # Root node
                    mean = self.models[node].expand(n_samples, 1)
                    noise = torch.randn_like(mean) * 0.1
                    sample_dict[node] = mean + noise
                    
                # Place in output tensor
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
                    # Hard intervention: set value directly
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
                        mean = self.models[node].expand(n_samples, 1)
                        noise = torch.randn_like(mean) * 0.1
                        sample_dict[node] = mean + noise
                
                idx = self.node_to_idx[node]
                samples[:, idx] = sample_dict[node].squeeze()
                
        return samples

    def counterfactual(self, observation: torch.Tensor, intervention_dict: Dict[str, float]) -> torch.Tensor:
        """
        Abduction-Action-Prediction (Simplified).
        1. Abduction: Infer noise from observation (inverse of structural equation).
           Since our functions are Neural Networks (non-invertible easily), we approximate noise 
           as the residual: U = X - f(Parents).
        2. Action: Modify graph (do-intervention).
        3. Prediction: Re-compute values using inferred noise and intervention.
        """
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
                    pred_mean = self.models[node].expand(1, 1)
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
                        cf_dict[node] = mean + noises[node] # Add the abducted noise
                    else:
                        mean = self.models[node].expand(1, 1)
                        cf_dict[node] = mean + noises[node]
                
                idx = self.node_to_idx[node]
                cf_samples[:, idx] = cf_dict[node].squeeze()
        
        return cf_samples

