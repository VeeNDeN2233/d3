"""
Продвинутые архитектуры автоэнкодеров для детекции аномалий.

Варианты:
1. Bidirectional LSTM + Attention
2. Transformer-based
3. Hybrid CNN-LSTM
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BidirectionalLSTMAutoencoder(nn.Module):
    """
    Bidirectional LSTM автоэнкодер с attention механизмом.
    
    Архитектура:
    - Encoder: Bidirectional LSTM [256, 128, 64]
    - Attention: Multi-head attention
    - Decoder: LSTM [64, 128, 256, 75]
    """

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
        """
        Инициализация Bidirectional LSTM автоэнкодера.
        
        Args:
            input_size: Размер входного вектора
            sequence_length: Длина последовательности
            encoder_hidden_sizes: Размеры скрытых слоев энкодера
            decoder_hidden_sizes: Размеры скрытых слоев декодера
            latent_size: Размер латентного представления
            num_attention_heads: Количество голов attention
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.latent_size = latent_size
        
        # Encoder: Bidirectional LSTM
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
            # Bidirectional удваивает hidden_size
            prev_size = hidden_size * 2
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            embed_dim=prev_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        # Проекция в латентное пространство
        self.latent_proj = nn.Linear(prev_size, latent_size)
        
        # Decoder: LSTM
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
        
        # Output projection
        self.output_proj = nn.Linear(decoder_hidden_sizes[-1], input_size)
        
        logger.info(
            f"Инициализирован BidirectionalLSTMAutoencoder: "
            f"input={input_size}, seq_len={sequence_length}, "
            f"latent={latent_size}, encoder={encoder_hidden_sizes}, "
            f"decoder={decoder_hidden_sizes}, attention_heads={num_attention_heads}"
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Кодирование последовательности.
        
        Args:
            x: Входная последовательность (batch, seq_len, input_size)
        
        Returns:
            Tuple (latent, attention_weights)
        """
        # Проходим через bidirectional LSTM слои
        h = x
        for lstm in self.encoder_lstms:
            h, _ = lstm(h)
            # h shape: (batch, seq_len, hidden_size * 2)
        
        # Применяем attention
        attended, attention_weights = self.attention(h, h, h)
        # attended shape: (batch, seq_len, hidden_size * 2)
        
        # Берем последний кадр после attention
        latent = attended[:, -1, :]  # (batch, hidden_size * 2)
        
        # Проекция в латентное пространство
        latent = self.latent_proj(latent)  # (batch, latent_size)
        
        return latent, attention_weights

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Декодирование латентного представления.
        
        Args:
            latent: Латентное представление (batch, latent_size)
        
        Returns:
            Восстановленная последовательность (batch, seq_len, input_size)
        """
        # Расширяем латентное представление на всю последовательность
        latent_expanded = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # latent_expanded shape: (batch, seq_len, latent_size)
        
        # Проходим через LSTM слои
        h = latent_expanded
        for lstm in self.decoder_lstms:
            h, _ = lstm(h)
        
        # Output projection
        reconstructed = self.output_proj(h)
        
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Входная последовательность (batch, seq_len, input_size)
        
        Returns:
            Tuple (reconstructed, latent)
        """
        latent, attention_weights = self.encode(x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based автоэнкодер для детекции аномалий.
    
    Использует self-attention для захвата долгосрочных зависимостей.
    """

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
        """
        Инициализация Transformer автоэнкодера.
        
        Args:
            input_size: Размер входного вектора
            sequence_length: Длина последовательности
            d_model: Размерность модели
            nhead: Количество голов attention
            num_encoder_layers: Количество слоев энкодера
            num_decoder_layers: Количество слоев декодера
            dim_feedforward: Размерность feedforward сети
            dropout: Dropout rate
            latent_size: Размер латентного представления
        """
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.latent_size = latent_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Latent projection
        self.latent_proj = nn.Linear(d_model, latent_size)
        
        # Latent expansion
        self.latent_expand = nn.Linear(latent_size, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, input_size)
        
        logger.info(
            f"Инициализирован TransformerAutoencoder: "
            f"input={input_size}, seq_len={sequence_length}, "
            f"d_model={d_model}, nhead={nhead}, "
            f"encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Кодирование последовательности."""
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        
        # Берем последний кадр
        latent = encoded[:, -1, :]  # (batch, d_model)
        
        # Проекция в латентное пространство
        latent = self.latent_proj(latent)  # (batch, latent_size)
        
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Декодирование латентного представления."""
        # Расширяем латентное представление
        latent_expanded = self.latent_expand(latent)  # (batch, d_model)
        latent_expanded = latent_expanded.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # (batch, seq_len, d_model)
        
        # Positional encoding для декодера
        latent_expanded = self.pos_encoder(latent_expanded)
        
        # Transformer decoder (self-attention)
        decoded = self.decoder(latent_expanded, latent_expanded)  # (batch, seq_len, d_model)
        
        # Output projection
        reconstructed = self.output_proj(decoded)  # (batch, seq_len, input_size)
        
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent


class PositionalEncoding(nn.Module):
    """Positional encoding для Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

