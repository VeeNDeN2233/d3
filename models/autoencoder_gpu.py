
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMAutoencoderGPU(nn.Module):

    def __init__(
        self,
        input_size: int = 69,
        sequence_length: int = 30,
        encoder_hidden_sizes: list = [128, 64, 32],
        decoder_hidden_sizes: list = [64, 128, 69],
        encoder_num_layers: int = 3,
        decoder_num_layers: int = 3,
        encoder_dropout: float = 0.2,
        decoder_dropout: float = 0.2,
        latent_size: int = 32,
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
                    dropout=encoder_dropout if i < len(encoder_hidden_sizes) - 1 else 0,
                )
            )
            prev_size = hidden_size
        

        self.latent_proj = nn.Linear(encoder_hidden_sizes[-1], latent_size)
        

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
        

        self.output_proj = nn.Linear(decoder_hidden_sizes[-1], input_size)
        
        logger.info(
            f"Инициализирован LSTM автоэнкодер: "
            f"input={input_size}, seq_len={sequence_length}, "
            f"latent={latent_size}, encoder={encoder_hidden_sizes}, "
            f"decoder={decoder_hidden_sizes}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        h = x
        for lstm in self.encoder_lstms:
            h, (hidden, cell) = lstm(h)

            h = hidden[-1].unsqueeze(1).expand(-1, h.size(1), -1)
        

        last_hidden = h[:, -1, :]
        

        latent = self.latent_proj(last_hidden)
        
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)
        


        decoder_input = latent.unsqueeze(1).expand(-1, self.sequence_length, -1)
        

        h = decoder_input
        for lstm in self.decoder_lstms:
            h, _ = lstm(h)
        

        output = self.output_proj(h)
        
        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        latent = self.encode(x)
        

        reconstructed = self.decode(latent)
        
        return reconstructed, latent

    def compute_reconstruction_loss(
        self, x: torch.Tensor, reconstructed: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return F.mse_loss(x, reconstructed, reduction=reduction)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def to_gpu(self) -> "LSTMAutoencoderGPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logger.info(f"Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU недоступен, используется CPU")
        return self.to(device)


class LSTMAutoencoderWithAttention(nn.Module):

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

            self.attention = nn.MultiheadAttention(
                embed_dim=latent_size, num_heads=4, batch_first=True
            )
            logger.info("Включен attention механизм")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_attention:

            latent = self.base_model.encode(x)

            latent_expanded = latent.unsqueeze(1)
            attended, _ = self.attention(latent_expanded, latent_expanded, latent_expanded)
            latent = attended.squeeze(1)
        else:
            latent = self.base_model.encode(x)
        
        reconstructed = self.base_model.decode(latent)
        
        return reconstructed, latent

