"""
LSTM автоэнкодер для детекции аномалий движений младенцев.

ТОЛЬКО для GPU с поддержкой mixed precision.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMAutoencoderGPU(nn.Module):
    """
    LSTM автоэнкодер ТОЛЬКО для GPU.
    
    Архитектура:
    Encoder: [batch, 30, 69] → LSTM(69→128) → LSTM(128→64) → LSTM(64→32)
    Decoder: [batch, 30, 32] → LSTM(32→64) → LSTM(64→128) → LSTM(128→69)
    """

    def __init__(
        self,
        input_size: int = 69,  # 23 сустава * 3 координаты
        sequence_length: int = 30,
        encoder_hidden_sizes: list = [128, 64, 32],
        decoder_hidden_sizes: list = [64, 128, 69],
        encoder_num_layers: int = 3,
        decoder_num_layers: int = 3,
        encoder_dropout: float = 0.2,
        decoder_dropout: float = 0.2,
        latent_size: int = 32,
    ):
        """
        Инициализация LSTM автоэнкодера.
        
        Args:
            input_size: Размер входного вектора (23 сустава * 3 координаты = 69)
            sequence_length: Длина последовательности (30 кадров)
            encoder_hidden_sizes: Размеры скрытых слоев энкодера
            decoder_hidden_sizes: Размеры скрытых слоев декодера
            encoder_num_layers: Количество слоев LSTM в энкодере
            decoder_num_layers: Количество слоев LSTM в декодере
            encoder_dropout: Dropout для энкодера
            decoder_dropout: Dropout для декодера
            latent_size: Размер латентного представления
        """
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.latent_size = latent_size
        
        # Encoder: последовательность → латентное представление
        self.encoder_lstms = nn.ModuleList()
        prev_size = input_size
        
        for i, hidden_size in enumerate(encoder_hidden_sizes):
            self.encoder_lstms.append(
                nn.LSTM(
                    input_size=prev_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=encoder_dropout if i < len(encoder_hidden_sizes) - 1 else 0,
                )
            )
            prev_size = hidden_size
        
        # Latent projection
        self.latent_proj = nn.Linear(encoder_hidden_sizes[-1], latent_size)
        
        # Decoder: латентное представление → последовательность
        self.decoder_lstms = nn.ModuleList()
        prev_size = latent_size
        
        for i, hidden_size in enumerate(decoder_hidden_sizes):
            self.decoder_lstms.append(
                nn.LSTM(
                    input_size=prev_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=decoder_dropout if i < len(decoder_hidden_sizes) - 1 else 0,
                )
            )
            prev_size = hidden_size
        
        # Output projection
        self.output_proj = nn.Linear(decoder_hidden_sizes[-1], input_size)
        
        logger.info(
            f"Инициализирован LSTM автоэнкодер: "
            f"input={input_size}, seq_len={sequence_length}, "
            f"latent={latent_size}, encoder={encoder_hidden_sizes}, "
            f"decoder={decoder_hidden_sizes}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Энкодер: последовательность → латентное представление.
        
        Args:
            x: Входной тензор формы (batch, sequence_length, input_size)
        
        Returns:
            Латентное представление формы (batch, latent_size)
        """
        # Проходим через все LSTM слои энкодера
        h = x
        for lstm in self.encoder_lstms:
            h, (hidden, cell) = lstm(h)
            # Используем последний hidden state
            h = hidden[-1].unsqueeze(1).expand(-1, h.size(1), -1)
        
        # Берем последний hidden state из последнего слоя
        last_hidden = h[:, -1, :]  # (batch, hidden_size)
        
        # Проекция в латентное пространство
        latent = self.latent_proj(last_hidden)  # (batch, latent_size)
        
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Декодер: латентное представление → последовательность.
        
        Args:
            latent: Латентное представление формы (batch, latent_size)
        
        Returns:
            Восстановленная последовательность формы (batch, sequence_length, input_size)
        """
        batch_size = latent.size(0)
        
        # Расширяем латентный вектор на всю длину последовательности
        # (batch, latent_size) → (batch, sequence_length, latent_size)
        decoder_input = latent.unsqueeze(1).expand(-1, self.sequence_length, -1)
        
        # Проходим через все LSTM слои декодера
        h = decoder_input
        for lstm in self.decoder_lstms:
            h, _ = lstm(h)
        
        # Финальная проекция в выходное пространство
        output = self.output_proj(h)  # (batch, sequence_length, input_size)
        
        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: кодирование и декодирование.
        
        Args:
            x: Входной тензор формы (batch, sequence_length, input_size)
        
        Returns:
            Tuple (reconstructed, latent):
            - reconstructed: Восстановленная последовательность
            - latent: Латентное представление
        """
        # Кодирование
        latent = self.encode(x)
        
        # Декодирование
        reconstructed = self.decode(latent)
        
        return reconstructed, latent

    def compute_reconstruction_loss(
        self, x: torch.Tensor, reconstructed: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Вычислить ошибку реконструкции (MSE).
        
        Args:
            x: Исходная последовательность
            reconstructed: Восстановленная последовательность
            reduction: "mean", "sum", или "none"
        
        Returns:
            Ошибка реконструкции
        """
        return F.mse_loss(x, reconstructed, reduction=reduction)

    def get_device(self) -> torch.device:
        """Получить устройство модели (GPU/CPU)."""
        return next(self.parameters()).device

    def to_gpu(self) -> "LSTMAutoencoderGPU":
        """Переместить модель на GPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logger.info(f"Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU недоступен, используется CPU")
        return self.to(device)


class LSTMAutoencoderWithAttention(nn.Module):
    """
    Расширенная версия с attention механизмом (опционально).
    """

    def __init__(
        self,
        input_size: int = 69,
        sequence_length: int = 30,
        encoder_hidden_sizes: list = [128, 64, 32],
        decoder_hidden_sizes: list = [64, 128, 69],
        latent_size: int = 32,
        use_attention: bool = False,
    ):
        super().__init__()
        
        self.base_model = LSTMAutoencoderGPU(
            input_size=input_size,
            sequence_length=sequence_length,
            encoder_hidden_sizes=encoder_hidden_sizes,
            decoder_hidden_sizes=decoder_hidden_sizes,
            latent_size=latent_size,
        )
        
        self.use_attention = use_attention
        
        if use_attention:
            # Attention механизм
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_size, num_heads=4, batch_first=True
            )
            logger.info("Включен attention механизм")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass с опциональным attention."""
        if self.use_attention:
            # Используем attention для улучшения кодирования
            latent = self.base_model.encode(x)
            # Применяем self-attention к латентному представлению
            latent_expanded = latent.unsqueeze(1)  # (batch, 1, latent_size)
            attended, _ = self.attention(latent_expanded, latent_expanded, latent_expanded)
            latent = attended.squeeze(1)  # (batch, latent_size)
        else:
            latent = self.base_model.encode(x)
        
        reconstructed = self.base_model.decode(latent)
        
        return reconstructed, latent

