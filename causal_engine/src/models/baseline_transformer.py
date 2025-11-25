import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class BaselineTransformer(nn.Module):
    """
    A standard Transformer model for tabular/time-series prediction.
    Treats features as a sequence or processes time-series.
    For tabular data with N features, we can embed each feature and treat as sequence of length N.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128):
        super().__init__()
        self.embedding = nn.Linear(1, d_model) # Treat each scalar feature as token
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model * input_dim, input_dim) # Flatten and predict
        self.input_dim = input_dim
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim]
        batch_size = x.shape[0]
        
        # Reshape to [input_dim, batch_size, 1] to treat features as sequence
        x_seq = x.permute(1, 0).unsqueeze(2) 
        
        # Embed: [input_dim, batch_size, d_model]
        x_emb = self.embedding(x_seq)
        x_emb = self.pos_encoder(x_emb)
        
        # Transform: [input_dim, batch_size, d_model]
        output = self.transformer_encoder(x_emb)
        
        # Flatten: [batch_size, input_dim * d_model]
        output = output.permute(1, 0, 2).reshape(batch_size, -1)
        
        # Predict reconstruction/value
        return self.decoder(output)

