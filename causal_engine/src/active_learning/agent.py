import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple
from ..causal_graph.discovery.base import CausalDiscoveryAlgorithm
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ActiveCausalAgent:
    """
    An agent that actively proposes interventions to refine the causal graph.
    Uses an ensemble of discovery algorithms (bootstrap) to estimate uncertainty.
    """
    
    def __init__(self, discovery_algo_class, config: Dict, n_ensemble: int = 5):
        self.discovery_class = discovery_algo_class
        self.config = config
        self.n_ensemble = n_ensemble
        self.graphs: List[nx.DiGraph] = []
        self.data_history: pd.DataFrame = None

    def update_belief(self, data: pd.DataFrame):
        """
        Refits the ensemble on the new dataset (history + new batch).
        """
        self.data_history = data if self.data_history is None else pd.concat([self.data_history, data])
        self.graphs = []
        
        logger.info(f"Updating active agent beliefs with {len(self.data_history)} samples...")
        
        # Bootstrap bagging
        for i in range(self.n_ensemble):
            # Sample with replacement
            bootstrap_data = self.data_history.sample(frac=1.0, replace=True, random_state=i)
            algo = self.discovery_class(self.config)
            algo.fit(bootstrap_data)
            self.graphs.append(algo.get_graph())
            
        logger.info("Belief update complete.")

    def propose_intervention(self) -> str:
        """
        Identifies the most uncertain edge and proposes an intervention on the parent.
        Returns the name of the node to intervene on.
        """
        if not self.graphs:
            raise ValueError("No beliefs yet. Call update_belief first.")
            
        # 1. Count edge frequencies
        edge_counts = {}
        nodes = self.graphs[0].nodes()
        possible_edges = [(u, v) for u in nodes for v in nodes if u != v]
        
        for u, v in possible_edges:
            count = sum(1 for G in self.graphs if G.has_edge(u, v))
            edge_counts[(u, v)] = count
            
        # 2. Find max entropy (uncertainty) -> count closest to n_ensemble/2
        target_count = self.n_ensemble / 2.0
        best_edge = None
        min_dist = float('inf')
        
        for edge, count in edge_counts.items():
            # Ignore edges that are almost certain (0 or n)
            if count == 0 or count == self.n_ensemble:
                continue
                
            dist = abs(count - target_count)
            if dist < min_dist:
                min_dist = dist
                best_edge = edge
                
        if best_edge:
            u, v = best_edge
            logger.info(f"Uncertainty found on edge {u}->{v} (Frequency: {edge_counts[best_edge]}/{self.n_ensemble}).")
            logger.info(f"Proposing intervention on parent node: {u}")
            return u
        else:
            logger.info("Graph structure is stable across ensemble. No critical interventions needed.")
            return None

    def get_consensus_graph(self, threshold: float = 0.5) -> nx.DiGraph:
        """
        Returns the consensus graph from the ensemble.
        """
        if not self.graphs:
            return nx.DiGraph()
            
        nodes = self.graphs[0].nodes()
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        
        possible_edges = [(u, v) for u in nodes for v in nodes if u != v]
        for u, v in possible_edges:
            count = sum(1 for g in self.graphs if g.has_edge(u, v))
            if count / self.n_ensemble >= threshold:
                G.add_edge(u, v)
                
        return G

