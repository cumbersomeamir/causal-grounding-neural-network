import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm
from ..data.loaders import DataLoader, split_data
from ..data.preprocess import Preprocessor
from ..causal_graph.structure_learning import learn_graph_from_data, save_graph, load_graph
from ..causal_graph.discovery.notears import DeepCausalDiscovery
from ..causal_graph.scm import SCM
from ..utils.logger import get_logger
from ..utils.seed import set_seed

logger = get_logger(__name__)

def train(data_path: str, graph_path: str = None, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3, algo: str = 'pc'):
    set_seed()
    
    # 1. Data Loading
    logger.info(f"Loading data from {data_path}")
    loader = DataLoader(data_path)
    raw_data = loader.get_data()
    
    # Preprocessing
    preprocessor = Preprocessor()
    data = preprocessor.fit_transform(raw_data)
    
    train_df, val_df, test_df = split_data(data)
    train_tensor = torch.tensor(train_df.values).float()
    val_tensor = torch.tensor(val_df.values).float()

    # 2. Structure Learning
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    if graph_path and os.path.exists(graph_path):
        logger.info(f"Loading graph from {graph_path}")
        graph = load_graph(graph_path)
    else:
        logger.info(f"Learning graph structure using {algo}...")
        if algo == 'notears':
            # Use the production-grade Deep Discovery
            discovery = DeepCausalDiscovery({'hidden_dim': 64, 'max_iter': 100})
            discovery.fit(train_df)
            graph = discovery.get_graph()
        else:
            # Use Standard PC
            graph = learn_graph_from_data(train_df)
            
        if graph_path:
            save_graph(graph, graph_path)

    # 3. Initialize SCM
    scm = SCM(graph)
    optimizer = optim.Adam(scm.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 4. Training Loop
    logger.info("Starting SCM training...")
    for epoch in range(epochs):
        scm.train()
        permutation = torch.randperm(train_tensor.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = train_tensor[indices]
            
            optimizer.zero_grad()
            # Forward pass: predict each node from its parents
            output = scm(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation
        scm.eval()
        with torch.no_grad():
            val_output = scm(val_tensor)
            val_loss = criterion(val_output, val_tensor).item()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss/len(permutation):.4f} - Val Loss: {val_loss:.4f}")

    # 5. Save Model
    os.makedirs("artifacts", exist_ok=True)
    torch.save(scm.state_dict(), "artifacts/scm_model.pth")
    with open("artifacts/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Training complete. Model saved to artifacts/scm_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--graph_path", type=str, default="artifacts/graph.pkl", help="Path to save/load graph")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--algo", type=str, default="pc", choices=["pc", "notears"], help="Discovery algorithm")
    args = parser.parse_args()
    
    train(args.data_path, args.graph_path, args.epochs, algo=args.algo)
