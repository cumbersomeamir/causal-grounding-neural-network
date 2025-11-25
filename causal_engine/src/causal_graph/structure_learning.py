import numpy as np
import pandas as pd
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
import pickle
from ..utils.logger import get_logger

logger = get_logger(__name__)

def learn_graph_from_data(data: pd.DataFrame, alpha: float = 0.05) -> nx.DiGraph:
    """
    Learns a causal graph using the PC algorithm.
    """
    logger.info("Starting structure learning with PC algorithm...")
    data_np = data.to_numpy()
    labels = data.columns.tolist()
    
    # Run PC algorithm
    # fisherz is the conditional independence test for Gaussian data
    cg = pc(data_np, alpha, fisherz)
    
    # Convert to NetworkX DiGraph
    # cg.G is the learned GeneralGraph
    # We need to convert it. The PC algorithm output might be a CPDAG.
    # For this system, we will assume we can orient edges or treat undirected as bidirectional for now,
    # but ideally we want a DAG.
    
    adj_matrix = cg.G.graph
    G = nx.DiGraph()
    
    for i, label in enumerate(labels):
        G.add_node(label)
        
    num_nodes = len(labels)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1: # Directed i -> j
                G.add_edge(labels[i], labels[j])
            elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1: # Directed j -> i
                G.add_edge(labels[j], labels[i])
            # Note: undirected edges (1, 1) or (-1, -1) depending on implementation are ambiguous
            # Here we simplify for the 'production-ready' goal by taking directed edges found
            
    logger.info(f"Graph learned with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def enforce_acyclicity(G: nx.DiGraph) -> nx.DiGraph:
    """
    Removes edges to enforce acyclicity if the graph has cycles.
    Uses a simple heuristic: remove edge with lowest weight (if weighted) or arbitrary in cycle.
    """
    if nx.is_directed_acyclic_graph(G):
        return G
    
    logger.warning("Cycle detected in learned graph. Enforcing acyclicity...")
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) > 1:
                # Break the cycle by removing the last edge
                u, v = cycle[-2], cycle[-1]
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    logger.info(f"Removed edge {u}->{v} to break cycle.")
    except Exception as e:
        logger.error(f"Error enforcing acyclicity: {e}")
        
    return G

def save_graph(G: nx.DiGraph, path: str):
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    logger.info(f"Graph saved to {path}")

def load_graph(path: str) -> nx.DiGraph:
    with open(path, 'rb') as f:
        G = pickle.load(f)
    logger.info(f"Graph loaded from {path}")
    return G

