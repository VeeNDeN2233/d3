
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BidirectionalLSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size: int = 75,
        sequence_length: int = 30,
        encoder_hidden_sizes: list = [256, 128, 64],
        decoder_hidden_sizes: list = [64, 128, 256, 75],
        latent_size: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.latent_size = latent_size
        

        self.encoder_lstms = nn.ModuleList()
        prev_size = input_size
        
        for i, hidden_size in enumerate(encoder_hidden_sizes):
            self.encoder_lstms.append(
                nn.LSTM(
                    input_size=prev_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if i < len(encoder_hidden_sizes) - 1 else 0,
                )
            )

            prev_size = hidden_size * 2
        

        self.attention = nn.MultiheadAttention(
            embed_dim=prev_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout,
        )
        

        self.latent_proj = nn.Linear(prev_size, latent_size)
        

        self.decoder_lstms = nn.ModuleList()
        prev_size = latent_size
        
        for i, hidden_size in enumerate(decoder_hidden_sizes):
            self.decoder_lstms.append(
                nn.LSTM(
                    input_size=prev_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout if i < len(decoder_hidden_sizes) - 1 else 0,
                )
            )
            prev_size = hidden_size
        

        self.output_proj = nn.Linear(decoder_hidden_sizes[-1], input_size)
        
        logger.info(
            f"Инициализирован BidirectionalLSTMAutoencoder: "
            f"input={input_size}, seq_len={sequence_length}, "
            f"latent={latent_size}, encoder={encoder_hidden_sizes}, "
            f"decoder={decoder_hidden_sizes}, attention_heads={num_attention_heads}"
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h = x
        for lstm in self.encoder_lstms:
            h, _ = lstm(h)

        

        attended, attention_weights = self.attention(h, h, h)

        

        latent = attended[:, -1, :]
        

        latent = self.latent_proj(latent)
        
        return latent, attention_weights

    def decode(self, latent: torch.Tensor) -> torch.Tensor:

        latent_expanded = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)

        

        h = latent_expanded
        for lstm in self.decoder_lstms:
            h, _ = lstm(h)
        

        reconstructed = self.output_proj(h)
        
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent, attention_weights = self.encode(x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent


class TransformerAutoencoder(nn.Module):

    def __init__(
        self,
        input_size: int = 75,
        sequence_length: int = 30,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_size: int = 128,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.latent_size = latent_size
        

        self.input_proj = nn.Linear(input_size, d_model)
        

        self.pos_encoder = PositionalEncoding(d_model, dropout, sequence_length)
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        

        self.latent_proj = nn.Linear(d_model, latent_size)
        

        self.latent_expand = nn.Linear(latent_size, d_model)
        

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        

        self.output_proj = nn.Linear(d_model, input_size)
        
        logger.info(
            f"Инициализирован TransformerAutoencoder: "
            f"input={input_size}, seq_len={sequence_length}, "
            f"d_model={d_model}, nhead={nhead}, "
            f"encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_proj(x)
        

        x = self.pos_encoder(x)
        

        encoded = self.encoder(x)
        

        latent = encoded[:, -1, :]
        

        latent = self.latent_proj(latent)
        
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:

        latent_expanded = self.latent_expand(latent)
        latent_expanded = latent_expanded.unsqueeze(1).repeat(1, self.sequence_length, 1)

        

        latent_expanded = self.pos_encoder(latent_expanded)
        

        decoded = self.decoder(latent_expanded, latent_expanded)
        

        reconstructed = self.output_proj(decoded)
        
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

