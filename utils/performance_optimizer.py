"""
Модуль оптимизации производительности.
Включает кэширование, батчинг и оптимизацию памяти.
"""

import functools
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU кэш с ограничением размера."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key][0]
        return None
    
    def set(self, key: str, value: Any):
        """Установить значение в кэш."""
        if len(self.cache) >= self.max_size:
            # Удаляем наименее используемый элемент
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = (value, time.time())
        self.access_times[key] = time.time()
    
    def clear(self):
        """Очистить кэш."""
        self.cache.clear()
        self.access_times.clear()


def cache_result(max_size: int = 100, ttl: Optional[float] = None):
    """
    Декоратор для кэширования результатов функций.
    
    Args:
        max_size: Максимальный размер кэша
        ttl: Время жизни кэша в секундах (None = без ограничения)
    """
    cache = LRUCache(max_size=max_size)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Создаем ключ кэша из аргументов
            cache_key = _create_cache_key(func.__name__, args, kwargs)
            
            # Проверяем кэш
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                if ttl is None or (time.time() - cache.access_times.get(cache_key, 0)) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
            
            # Выполняем функцию
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        wrapper.cache_clear = cache.clear
        return wrapper
    
    return decorator


def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Создать ключ кэша из аргументов функции."""
    # Преобразуем аргументы в строку
    key_parts = [func_name]
    
    # Обрабатываем args
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, Path):
            key_parts.append(str(arg))
        elif isinstance(arg, np.ndarray):
            # Используем хеш массива
            key_parts.append(hashlib.md5(arg.tobytes()).hexdigest()[:16])
        else:
            key_parts.append(str(type(arg).__name__))
    
    # Обрабатываем kwargs
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        elif isinstance(v, Path):
            key_parts.append(f"{k}={str(v)}")
        elif isinstance(v, np.ndarray):
            key_parts.append(f"{k}={hashlib.md5(v.tobytes()).hexdigest()[:16]}")
    
    return ":".join(key_parts)


def batch_process(
    items: List[Any],
    batch_size: int,
    process_fn: Callable[[List[Any]], List[Any]],
    use_gpu: bool = True
) -> List[Any]:
    """
    Обработка элементов батчами для оптимизации производительности.
    
    Args:
        items: Список элементов для обработки
        batch_size: Размер батча
        process_fn: Функция обработки батча
        use_gpu: Использовать ли GPU для обработки
    
    Returns:
        Список обработанных элементов
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        if use_gpu and torch.cuda.is_available():
            # Обрабатываем батч на GPU
            batch_results = process_fn(batch)
        else:
            batch_results = process_fn(batch)
        
        results.extend(batch_results)
        
        # Очистка памяти GPU после каждого батча
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def optimize_memory():
    """Оптимизация использования памяти."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()


class PerformanceMonitor:
    """Мониторинг производительности функций."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    def time_function(self, func_name: str):
        """Декоратор для измерения времени выполнения функции."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if func_name not in self.timings:
                    self.timings[func_name] = []
                self.timings[func_name].append(elapsed)
                
                logger.debug(f"{func_name} took {elapsed:.4f}s")
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Получить статистику производительности."""
        stats = {}
        for func_name, timings in self.timings.items():
            if timings:
                stats[func_name] = {
                    "mean": np.mean(timings),
                    "std": np.std(timings),
                    "min": np.min(timings),
                    "max": np.max(timings),
                    "count": len(timings)
                }
        return stats
    
    def reset(self):
        """Сбросить статистику."""
        self.timings.clear()


# Глобальный монитор производительности
_performance_monitor = PerformanceMonitor()


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Получить статистику производительности."""
    return _performance_monitor.get_stats()


def reset_performance_stats():
    """Сбросить статистику производительности."""
    _performance_monitor.reset()

