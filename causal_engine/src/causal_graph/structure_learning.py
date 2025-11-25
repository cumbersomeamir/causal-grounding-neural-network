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
    cg = pc(data_np, alpha, fisherz)
    
    adj_matrix = cg.G.graph
    logger.info(f"Adjacency Matrix:\n{adj_matrix}")
    logger.info(f"Matrix Type: {type(adj_matrix)}")
    logger.info(f"Element Type: {type(adj_matrix[0,0])}")
    
    G = nx.DiGraph()
    
    for i, label in enumerate(labels):
        G.add_node(label)
        
    num_nodes = len(labels)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            
            val_ij = adj_matrix[i, j]
            val_ji = adj_matrix[j, i]
            
            # i -> j: tail at i (-1), arrow at j (1)
            if val_ij == -1 and val_ji == 1: 
                logger.info(f"Directed Edge {labels[i]} -> {labels[j]}")
                G.add_edge(labels[i], labels[j])
            # j -> i: arrow at i (1), tail at j (-1)
            elif val_ij == 1 and val_ji == -1: 
                logger.info(f"Directed Edge {labels[j]} -> {labels[i]}")
                G.add_edge(labels[j], labels[i])
            # Undirected i -- j: tail at i (-1), tail at j (-1)
            # OR arrow at i (1), arrow at j (1) - depends on PC implementation
            elif (val_ij == -1 and val_ji == -1) or (val_ij == 1 and val_ji == 1):
                logger.info(f"Undirected Edge {labels[i]} -- {labels[j]} (vals: {val_ij}, {val_ji})")
                # Heuristic: Orient i -> j if i < j to avoid cycles and double edges
                if i < j:
                    G.add_edge(labels[i], labels[j])
                    logger.info(f"Orienting {labels[i]} -> {labels[j]}")
                else:
                     # If we already added j -> i (when we were at j, i), good.
                     # If i > j, we rely on the j loop to have added j -> i.
                     pass
            
    logger.info(f"Graph learned with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    G = enforce_acyclicity(G)
    return G

def enforce_acyclicity(G: nx.DiGraph) -> nx.DiGraph:
    """
    Removes edges to enforce acyclicity if the graph has cycles.
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
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    logger.info(f"Graph saved to {path}")

def load_graph(path: str) -> nx.DiGraph:
    with open(path, 'rb') as f:
        G = pickle.load(f)
    logger.info(f"Graph loaded from {path}")
    return G
