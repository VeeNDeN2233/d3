"""
Бенчмарк производительности системы детекции аномалий.

Измеряет скорость инференса при различных batch sizes и настройках.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from models.anomaly_detector import AnomalyDetector
from models.autoencoder_gpu import LSTMAutoencoderGPU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_inference(
    model: LSTMAutoencoderGPU,
    detector: AnomalyDetector,
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    sequence_length: int = 30,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> List[Dict]:
    """
    Бенчмарк скорости инференса.
    
    Args:
        model: Модель автоэнкодера
        detector: Детектор аномалий
        batch_sizes: Список размеров батчей для тестирования
        sequence_length: Длина последовательности
        num_iterations: Количество итераций для измерения
        warmup_iterations: Количество итераций разогрева
    
    Returns:
        Список результатов бенчмарка
    """
    device = model.get_device()
    input_size = model.input_size
    
    results = []
    
    logger.info("=" * 60)
    logger.info("БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    logger.info("=" * 60)
    logger.info(f"Устройство: {device}")
    logger.info(f"Input size: {input_size}, Sequence length: {sequence_length}")
    logger.info(f"Iterations: {num_iterations}, Warmup: {warmup_iterations}")
    logger.info("")
    
    for batch_size in batch_sizes:
        logger.info(f"Тестирование batch_size={batch_size}...")
        
        # Создаем тестовый тензор
        test_data = torch.randn(batch_size, sequence_length, input_size).to(device)
        
        # Разогрев
        model.eval()
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(test_data)
        
        # Синхронизация GPU
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Измеряем время инференса модели
        start_time = time.time()
        
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            for _ in range(num_iterations):
                reconstructed, latent = model(test_data)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        model_time = time.time() - start_time
        
        # Измеряем время детекции аномалий
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                is_anomaly, errors = detector.predict(test_data)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        detector_time = time.time() - start_time
        
        # Вычисляем метрики
        total_frames = batch_size * num_iterations
        total_time = model_time + detector_time
        
        fps = total_frames / total_time
        ms_per_frame = 1000 / fps if fps > 0 else float("inf")
        ms_per_batch = total_time / num_iterations * 1000
        
        # Использование памяти GPU
        gpu_memory = None
        if device.type == "cuda":
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_memory = {
                "allocated_gb": round(gpu_memory_allocated, 2),
                "reserved_gb": round(gpu_memory_reserved, 2),
            }
        
        result = {
            "batch_size": batch_size,
            "model_time_sec": round(model_time, 4),
            "detector_time_sec": round(detector_time, 4),
            "total_time_sec": round(total_time, 4),
            "fps": round(fps, 2),
            "ms_per_frame": round(ms_per_frame, 2),
            "ms_per_batch": round(ms_per_batch, 2),
            "gpu_memory": gpu_memory,
        }
        
        results.append(result)
        
        logger.info(
            f"  Batch {batch_size}: {fps:.1f} FPS, "
            f"{ms_per_frame:.1f} ms/frame, "
            f"{ms_per_batch:.1f} ms/batch"
        )
        if gpu_memory:
            logger.info(
                f"  GPU Memory: {gpu_memory['allocated_gb']:.2f} GB allocated, "
                f"{gpu_memory['reserved_gb']:.2f} GB reserved"
            )
    
    return results


def benchmark_throughput(
    model: LSTMAutoencoderGPU,
    detector: AnomalyDetector,
    total_sequences: int = 1000,
    batch_size: int = 32,
) -> Dict:
    """
    Бенчмарк пропускной способности (throughput).
    
    Args:
        model: Модель автоэнкодера
        detector: Детектор аномалий
        total_sequences: Общее количество последовательностей для обработки
        batch_size: Размер батча
    
    Returns:
        Словарь с результатами
    """
    device = model.get_device()
    input_size = model.input_size
    sequence_length = model.sequence_length
    
    logger.info("\n" + "=" * 60)
    logger.info("БЕНЧМАРК ПРОПУСКНОЙ СПОСОБНОСТИ")
    logger.info("=" * 60)
    logger.info(f"Total sequences: {total_sequences}, Batch size: {batch_size}")
    
    # Создаем тестовые данные
    num_batches = (total_sequences + batch_size - 1) // batch_size
    test_data = torch.randn(batch_size, sequence_length, input_size).to(device)
    
    model.eval()
    
    # Разогрев
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_data)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    processed = 0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
        for i in range(num_batches):
            batch = test_data
            if i == num_batches - 1:
                # Последний батч может быть меньше
                remaining = total_sequences - processed
                if remaining < batch_size:
                    batch = test_data[:remaining]
            
            reconstructed, _ = model(batch)
            is_anomaly, errors = detector.predict(batch)
            
            processed += batch.size(0)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    throughput = processed / total_time
    
    logger.info(f"Processed: {processed} sequences in {total_time:.2f} seconds")
    logger.info(f"Throughput: {throughput:.2f} sequences/second")
    logger.info(f"Time per sequence: {1000 / throughput:.2f} ms")
    
    return {
        "total_sequences": processed,
        "total_time_sec": round(total_time, 2),
        "throughput_seq_per_sec": round(throughput, 2),
        "ms_per_sequence": round(1000 / throughput, 2),
    }


def main():
    """Запустить все бенчмарки."""
    
    # Проверяем GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.error("GPU недоступен! Бенчмарк требует GPU.")
        return
    
    logger.info(f"Используется устройство: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Загружаем конфигурацию
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Загружаем модель и детектор
    checkpoint_path = Path("checkpoints/anomaly_detector.pt")
    if not checkpoint_path.exists():
        logger.error("Checkpoint не найден! Сначала обучите модель.")
        return
    
    from inference_gpu import load_model_and_detector
    
    model, detector = load_model_and_detector(checkpoint_path, config, device)
    
    # Бенчмарк инференса
    inference_results = benchmark_inference(
        model,
        detector,
        batch_sizes=[1, 4, 8, 16, 32],
        sequence_length=config["pose"]["sequence_length"],
        num_iterations=100,
    )
    
    # Бенчмарк пропускной способности
    throughput_result = benchmark_throughput(
        model, detector, total_sequences=1000, batch_size=32
    )
    
    # Сохраняем результаты
    results = {
        "inference_benchmark": inference_results,
        "throughput_benchmark": throughput_result,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    
    results_path = Path("results/performance_benchmark.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nРезультаты сохранены: {results_path}")
    
    # Сводка
    logger.info("\n" + "=" * 60)
    logger.info("СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ")
    logger.info("=" * 60)
    
    best_fps = max(r["fps"] for r in inference_results)
    best_batch = max(inference_results, key=lambda x: x["fps"])
    
    logger.info(f"Лучшая производительность: {best_fps:.1f} FPS (batch_size={best_batch['batch_size']})")
    logger.info(f"Пропускная способность: {throughput_result['throughput_seq_per_sec']:.1f} seq/sec")


if __name__ == "__main__":
    main()

