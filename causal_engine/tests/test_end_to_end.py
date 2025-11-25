import pandas as pd
import numpy as np
from src.causal_graph.structure_learning import learn_graph_from_data
import networkx as nx

def test_learn_graph():
    # Generate simple causal data: A -> B
    np.random.seed(42)
    A = np.random.randn(100)
    B = 2 * A + np.random.randn(100) * 0.1
    df = pd.DataFrame({'A': A, 'B': B})
    
    G = learn_graph_from_data(df)
    # Should find A -> B
    assert G.has_edge('A', 'B')
    assert not G.has_edge('B', 'A')

