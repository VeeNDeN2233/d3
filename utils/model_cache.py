"""
Кэширование и оптимизация загрузки моделей.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

from models.anomaly_detector import AnomalyDetector
from models.autoencoder_advanced import BidirectionalLSTMAutoencoder

logger = logging.getLogger(__name__)


class ModelCache:
    """Кэш для моделей с потокобезопасностью."""
    
    def __init__(self):
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._device: Optional[torch.device] = None
    
    def get_device(self) -> torch.device:
        """Получить устройство (GPU/CPU) с кэшированием."""
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
    
    def get_model_key(self, checkpoint_path: Path, model_type: str) -> str:
        """Создать ключ для модели."""
        return f"{checkpoint_path}:{model_type}"
    
    def get(self, checkpoint_path: Path, model_type: str) -> Optional[Tuple[BidirectionalLSTMAutoencoder, AnomalyDetector]]:
        """Получить модель из кэша."""
        key = self.get_model_key(checkpoint_path, model_type)
        with self._lock:
            return self._cache.get(key)
    
    def set(self, checkpoint_path: Path, model_type: str, model: BidirectionalLSTMAutoencoder, detector: AnomalyDetector):
        """Сохранить модель в кэш."""
        key = self.get_model_key(checkpoint_path, model_type)
        with self._lock:
            self._cache[key] = (model, detector)
            logger.info(f"Модель закэширована: {key}")
    
    def clear(self):
        """Очистить кэш."""
        with self._lock:
            for model, detector in self._cache.values():
                del model
                del detector
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Кэш моделей очищен")
    
    def is_cached(self, checkpoint_path: Path, model_type: str) -> bool:
        """Проверить, закэширована ли модель."""
        key = self.get_model_key(checkpoint_path, model_type)
        with self._lock:
            return key in self._cache


# Глобальный кэш моделей
_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Получить глобальный кэш моделей."""
    return _model_cache

