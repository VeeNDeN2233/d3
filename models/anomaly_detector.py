
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold_percentile: float = 95.0,
    ):
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
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i : i + batch_size].to(self.device)
                

                reconstructed, _ = self.model(batch)
                

                mse = nn.functional.mse_loss(
                    batch, reconstructed, reduction="none"
                )
                mse_per_sequence = mse.mean(dim=(1, 2))
                
                errors.extend(mse_per_sequence.cpu().numpy())
        
        return np.array(errors)

    def fit_threshold(self, val_sequences: torch.Tensor) -> float:
        logger.info("Вычисление порога аномалий на validation данных...")
        

        val_errors = self.compute_reconstruction_errors(val_sequences)
        self.val_errors = val_errors
        

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
        if self.threshold is None:
            raise ValueError(
                "Порог не установлен! Вызовите fit_threshold() на validation данных."
            )
        

        errors = self.compute_reconstruction_errors(sequences, batch_size)
        

        is_anomaly = errors > self.threshold
        
        return is_anomaly, errors

    def predict_proba(self, sequences: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("Порог не установлен!")
        
        errors = self.compute_reconstruction_errors(sequences, batch_size)
        



        proba = 1.0 - np.exp(-errors / self.threshold)
        proba = np.clip(proba, 0.0, 1.0)
        
        return proba

    def get_anomaly_score(self, sequences: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("Порог не установлен!")
        
        errors = self.compute_reconstruction_errors(sequences, batch_size)
        scores = errors / self.threshold
        
        return scores

    def save(self, save_path: Path) -> None:
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
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Файл не найден: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        

        detector = cls(
            model=model,
            device=device,
            threshold_percentile=checkpoint.get("threshold_percentile", 95.0),
        )
        

        detector.threshold = checkpoint.get("threshold")
        detector.val_errors = checkpoint.get("val_errors")
        
        logger.info(f"Детектор загружен: {load_path}")
        logger.info(f"Порог: {detector.threshold}")
        
        return detector

    def get_statistics(self) -> Dict[str, float]:
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

