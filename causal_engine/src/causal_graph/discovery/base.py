from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional

class CausalDiscoveryAlgorithm(ABC):
    """
    Abstract Base Class for Causal Structure Learning algorithms.
    Enforces a common interface for production pipelines.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._graph: Optional[nx.DiGraph] = None

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'CausalDiscoveryAlgorithm':
        """
        Fits the structure learning algorithm to the data.
        """
        pass

    @abstractmethod
    def get_graph(self) -> nx.DiGraph:
        """
        Returns the learned causal graph.
        """
        pass

    def save(self, path: str):
        """
        Saves the learned graph state.
        """
        import pickle
        import os
        if self._graph is None:
            raise ValueError("Graph has not been learned yet. Call fit() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._graph, f)

    def load(self, path: str):
        """
        Loads a graph state.
        """
        import pickle
        with open(path, 'rb') as f:
            self._graph = pickle.load(f)

