import torch
import torch.optim as optim
import argparse
import os
import pickle # Added missing import
from ..data.loaders import DataLoader, split_data
from ..data.preprocess import Preprocessor
from ..models.baseline_transformer import BaselineTransformer
from ..utils.logger import get_logger
from ..utils.seed import set_seed

logger = get_logger(__name__)

def train(data_path: str, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    set_seed()
    
    # 1. Data Loading
    loader = DataLoader(data_path)
    raw_data = loader.get_data()
    
    preprocessor = Preprocessor()
    data = preprocessor.fit_transform(raw_data)
    
    train_df, val_df, _ = split_data(data)
    train_tensor = torch.tensor(train_df.values).float()
    val_tensor = torch.tensor(val_df.values).float()
    
    input_dim = train_tensor.shape[1]

    # 2. Initialize Baseline
    model = BaselineTransformer(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 3. Training Loop
    logger.info("Starting Baseline training...")
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_tensor.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = train_tensor[indices]
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_tensor)
            val_loss = criterion(val_output, val_tensor).item()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss/len(permutation):.4f} - Val Loss: {val_loss:.4f}")

    # 4. Save Model
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/baseline_model.pth")
    logger.info("Baseline training complete. Model saved to artifacts/baseline_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train(args.data_path, args.epochs)

