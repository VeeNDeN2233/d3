"""
Детектор аномалий движений младенцев.

С защитой от утечки данных: threshold вычисляется ТОЛЬКО на validation данных.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Детектор аномалий с защитой от утечки данных:
    1. threshold вычисляется ТОЛЬКО на validation данных
    2. Никакие test данные не используются при обучении
    3. Сохранение порога вместе с моделью
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold_percentile: float = 95.0,
    ):
        """
        Инициализация детектора аномалий.
        
        Args:
            model: Обученный LSTM автоэнкодер
            device: Устройство (GPU/CPU)
            threshold_percentile: Перцентиль для вычисления порога (95 = 95-й перцентиль)
        """
        self.model = model
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None
        self.val_errors: Optional[np.ndarray] = None
        
        logger.info(
            f"Инициализирован AnomalyDetector: "
            f"device={device}, threshold_percentile={threshold_percentile}"
        )

    def compute_reconstruction_errors(
        self, sequences: torch.Tensor, batch_size: int = 32
    ) -> np.ndarray:
        """
        Вычислить ошибки реконструкции для последовательностей.
        
        Args:
            sequences: Тензор последовательностей (N, seq_len, input_size)
            batch_size: Размер батча для обработки
        
        Returns:
            Массив ошибок реконструкции (N,)
        """
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i : i + batch_size].to(self.device)
                
                # Forward pass
                reconstructed, _ = self.model(batch)
                
                # Вычисляем MSE для каждой последовательности
                mse = nn.functional.mse_loss(
                    batch, reconstructed, reduction="none"
                )  # (batch, seq_len, input_size)
                mse_per_sequence = mse.mean(dim=(1, 2))  # (batch,)
                
                errors.extend(mse_per_sequence.cpu().numpy())
        
        return np.array(errors)

    def fit_threshold(self, val_sequences: torch.Tensor) -> float:
        """
        Вычислить порог аномалий ТОЛЬКО на validation данных.
        
        ВАЖНО: Этот метод должен вызываться ТОЛЬКО с validation данными,
        НИКОГДА с test данными!
        
        Args:
            val_sequences: Validation последовательности (N, seq_len, input_size)
        
        Returns:
            Порог аномалий
        """
        logger.info("Вычисление порога аномалий на validation данных...")
        
        # Вычисляем ошибки реконструкции на validation данных
        val_errors = self.compute_reconstruction_errors(val_sequences)
        self.val_errors = val_errors
        
        # Вычисляем порог как перцентиль ошибок
        threshold = np.percentile(val_errors, self.threshold_percentile)
        self.threshold = float(threshold)
        
        logger.info(
            f"Порог аномалий установлен: {threshold:.6f} "
            f"(percentile {self.threshold_percentile} из {len(val_errors)} validation ошибок)"
        )
        logger.info(
            f"Статистика ошибок: min={val_errors.min():.6f}, "
            f"max={val_errors.max():.6f}, mean={val_errors.mean():.6f}, "
            f"std={val_errors.std():.6f}"
        )
        
        return self.threshold

    def predict(
        self, sequences: torch.Tensor, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказать аномалии для последовательностей.
        
        Args:
            sequences: Тензор последовательностей (N, seq_len, input_size)
            batch_size: Размер батча
        
        Returns:
            Tuple (is_anomaly, errors):
            - is_anomaly: Булев массив (N,) - True если аномалия
            - errors: Массив ошибок реконструкции (N,)
        """
        if self.threshold is None:
            raise ValueError(
                "Порог не установлен! Вызовите fit_threshold() на validation данных."
            )
        
        # Вычисляем ошибки реконструкции
        errors = self.compute_reconstruction_errors(sequences, batch_size)
        
        # Определяем аномалии
        is_anomaly = errors > self.threshold
        
        return is_anomaly, errors

    def predict_proba(self, sequences: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Предсказать вероятность аномалии (нормализованные ошибки).
        
        Args:
            sequences: Тензор последовательностей
            batch_size: Размер батча
        
        Returns:
            Массив вероятностей аномалии (N,) в диапазоне [0, 1]
        """
        if self.threshold is None:
            raise ValueError("Порог не установлен!")
        
        errors = self.compute_reconstruction_errors(sequences, batch_size)
        
        # Нормализуем ошибки относительно порога
        # Ошибка = threshold → вероятность = 0.5
        # Ошибка >> threshold → вероятность → 1.0
        proba = 1.0 - np.exp(-errors / self.threshold)
        proba = np.clip(proba, 0.0, 1.0)
        
        return proba

    def get_anomaly_score(self, sequences: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Получить score аномалии (нормализованные ошибки относительно порога).
        
        Args:
            sequences: Тензор последовательностей
            batch_size: Размер батча
        
        Returns:
            Массив scores (N,), где score > 1.0 означает аномалию
        """
        if self.threshold is None:
            raise ValueError("Порог не установлен!")
        
        errors = self.compute_reconstruction_errors(sequences, batch_size)
        scores = errors / self.threshold
        
        return scores

    def save(self, save_path: Path) -> None:
        """
        Сохранить детектор (модель + порог).
        
        Args:
            save_path: Путь для сохранения
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "val_errors": self.val_errors,
            "model_config": {
                "input_size": getattr(self.model, "input_size", None),
                "sequence_length": getattr(self.model, "sequence_length", None),
                "latent_size": getattr(self.model, "latent_size", None),
            },
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Детектор сохранен: {save_path}")

    @classmethod
    def load(
        cls, load_path: Path, model: nn.Module, device: torch.device
    ) -> "AnomalyDetector":
        """
        Загрузить детектор (модель + порог).
        
        Args:
            load_path: Путь к сохраненному детектору
            model: Модель автоэнкодера (должна соответствовать сохраненной)
            device: Устройство
        
        Returns:
            Загруженный детектор
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Файл не найден: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        # Загружаем веса модели
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        # Создаем детектор
        detector = cls(
            model=model,
            device=device,
            threshold_percentile=checkpoint.get("threshold_percentile", 95.0),
        )
        
        # Восстанавливаем порог и ошибки
        detector.threshold = checkpoint.get("threshold")
        detector.val_errors = checkpoint.get("val_errors")
        
        logger.info(f"Детектор загружен: {load_path}")
        logger.info(f"Порог: {detector.threshold}")
        
        return detector

    def get_statistics(self) -> Dict[str, float]:
        """Получить статистику детектора."""
        stats = {
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
        }
        
        if self.val_errors is not None:
            stats.update({
                "val_errors_min": float(self.val_errors.min()),
                "val_errors_max": float(self.val_errors.max()),
                "val_errors_mean": float(self.val_errors.mean()),
                "val_errors_std": float(self.val_errors.std()),
                "val_errors_count": len(self.val_errors),
            })
        
        return stats

