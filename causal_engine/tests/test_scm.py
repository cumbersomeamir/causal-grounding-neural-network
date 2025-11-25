import pytest
import torch
import networkx as nx
from src.causal_graph.scm import SCM

def test_scm_initialization():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")
    
    scm = SCM(G)
    assert len(scm.models) == 2
    assert isinstance(scm.models["B"], torch.nn.Sequential)

def test_scm_forward():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")
    
    scm = SCM(G)
    x = torch.randn(10, 2)
    output = scm(x)
    assert output.shape == (10, 2)

def test_scm_sample():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")
    
    scm = SCM(G)
    samples = scm.sample(5)
    assert samples.shape == (5, 2)

def test_scm_intervention():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")
    
    scm = SCM(G)
    intervention = {"A": 5.0}
    samples = scm.intervene(intervention, 5)
    
    # Check if A is exactly 5.0 (approx for float)
    assert torch.allclose(samples[:, 0], torch.tensor(5.0))
    assert samples.shape == (5, 2)

