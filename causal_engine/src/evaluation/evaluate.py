import torch
import pandas as pd
import argparse
import pickle
import os
from ..data.loaders import DataLoader, split_data
from ..causal_graph.scm import SCM
from ..causal_graph.structure_learning import load_graph
from ..models.baseline_transformer import BaselineTransformer
from ..utils.metrics import compute_rmse
from ..utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(data_path: str, scm_path: str, baseline_path: str, graph_path: str):
    # Load Data
    loader = DataLoader(data_path)
    raw_data = loader.get_data()
    
    with open("artifacts/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    
    data = preprocessor.transform(raw_data)
    _, _, test_df = split_data(data)
    test_tensor = torch.tensor(test_df.values).float()
    
    # Load SCM
    graph = load_graph(graph_path)
    scm = SCM(graph)
    scm.load_state_dict(torch.load(scm_path))
    scm.eval()
    
    # Load Baseline
    input_dim = test_tensor.shape[1]
    baseline = BaselineTransformer(input_dim)
    baseline.load_state_dict(torch.load(baseline_path))
    baseline.eval()
    
    # Evaluate
    with torch.no_grad():
        scm_pred = scm(test_tensor)
        baseline_pred = baseline(test_tensor)
        
    scm_rmse = compute_rmse(test_tensor.numpy(), scm_pred.numpy())
    baseline_rmse = compute_rmse(test_tensor.numpy(), baseline_pred.numpy())
    
    logger.info("Evaluation Results:")
    logger.info(f"SCM RMSE: {scm_rmse:.4f}")
    logger.info(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    if scm_rmse < baseline_rmse:
        logger.info("Causal SCM outperformed the baseline!")
    else:
        logger.info("Baseline outperformed the SCM (Correlation > Causation in this setting?)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--scm_path", type=str, default="artifacts/scm_model.pth")
    parser.add_argument("--baseline_path", type=str, default="artifacts/baseline_model.pth")
    parser.add_argument("--graph_path", type=str, default="artifacts/graph.pkl")
    args = parser.parse_args()
    
    evaluate(args.data_path, args.scm_path, args.baseline_path, args.graph_path)

