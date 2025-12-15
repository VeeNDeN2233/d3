"""
Тест модели на случайном шуме.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from models.anomaly_detector import AnomalyDetector
from models.autoencoder_advanced import BidirectionalLSTMAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_random_noise():
    """Тест модели на случайном шуме."""
    logger.info("=" * 60)
    logger.info("ТЕСТ МОДЕЛИ НА СЛУЧАЙНОМ ШУМЕ")
    logger.info("=" * 60)
    
    # Загружаем конфигурацию
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем модель
    checkpoint_path = Path("checkpoints/best_model_advanced.pt")
    if not checkpoint_path.exists():
        logger.error("Checkpoint не найден!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = BidirectionalLSTMAutoencoder(
        input_size=config["model"]["input_size"],
        sequence_length=config["pose"]["sequence_length"],
        encoder_hidden_sizes=[256, 128, 64],
        decoder_hidden_sizes=[64, 128, 256, 75],
        latent_size=64,
        num_attention_heads=4,
        dropout=0.2,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Загружаем детектор
    detector_path = Path("checkpoints/anomaly_detector_advanced.pt")
    detector = AnomalyDetector.load(detector_path, model, device)
    
    logger.info(f"Модель загружена. Порог: {detector.threshold:.6f}")
    
    # Тестируем разные типы шума
    test_cases = [
        {
            "name": "Случайный шум (равномерное распределение)",
            "data": lambda: np.random.uniform(-1, 1, (10, 30, 75)),
        },
        {
            "name": "Случайный шум (нормальное распределение)",
            "data": lambda: np.random.normal(0, 1, (10, 30, 75)),
        },
        {
            "name": "Нули (пустые данные)",
            "data": lambda: np.zeros((10, 30, 75)),
        },
        {
            "name": "Единицы (константа)",
            "data": lambda: np.ones((10, 30, 75)),
        },
        {
            "name": "Случайный шум (малый диапазон)",
            "data": lambda: np.random.uniform(-0.1, 0.1, (10, 30, 75)),
        },
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Тест: {test_case['name']}")
        logger.info(f"{'='*60}")
        
        # Генерируем тестовые данные
        test_data = test_case["data"]()
        test_tensor = torch.FloatTensor(test_data).to(device)
        
        # Вычисляем ошибку реконструкции
        with torch.no_grad(), torch.cuda.amp.autocast():
            errors = detector.compute_reconstruction_errors(test_tensor, batch_size=32)
        
        mean_error = float(errors.mean())
        max_error = float(errors.max())
        min_error = float(errors.min())
        std_error = float(errors.std())
        
        # Определяем аномалии
        is_anomaly = errors > detector.threshold
        anomaly_rate = float(is_anomaly.mean() * 100)
        
        logger.info(f"  Mean error: {mean_error:.6f}")
        logger.info(f"  Min error: {min_error:.6f}")
        logger.info(f"  Max error: {max_error:.6f}")
        logger.info(f"  Std error: {std_error:.6f}")
        logger.info(f"  Threshold: {detector.threshold:.6f}")
        logger.info(f"  Anomaly rate: {anomaly_rate:.1f}%")
        logger.info(f"  Ratio (error/threshold): {mean_error / detector.threshold:.2f}x")
        
        if mean_error > detector.threshold:
            logger.info(f"  ✅ Шум правильно детектируется как аномалия")
        else:
            logger.warning(f"  ⚠️ Шум НЕ детектируется как аномалия (ошибка ниже порога)")
        
        results.append({
            "name": test_case["name"],
            "mean_error": mean_error,
            "max_error": max_error,
            "min_error": min_error,
            "std_error": std_error,
            "anomaly_rate": anomaly_rate,
            "is_anomaly_detected": mean_error > detector.threshold,
        })
    
    # Сравнение с реальным видео
    logger.info(f"\n{'='*60}")
    logger.info("СРАВНЕНИЕ С РЕАЛЬНЫМ ВИДЕО")
    logger.info(f"{'='*60}")
    
    real_video_error = 0.006339  # Из medical_report.json
    logger.info(f"Реальное видео (baby.mp4): {real_video_error:.6f}")
    logger.info(f"Порог: {detector.threshold:.6f}")
    logger.info(f"Отношение: {real_video_error / detector.threshold:.2f}x")
    
    # Выводы
    logger.info(f"\n{'='*60}")
    logger.info("ВЫВОДЫ")
    logger.info(f"{'='*60}")
    
    noise_errors = [r["mean_error"] for r in results if "шум" in r["name"].lower()]
    if noise_errors:
        avg_noise_error = np.mean(noise_errors)
        logger.info(f"Средняя ошибка на шуме: {avg_noise_error:.6f}")
        logger.info(f"Ошибка на реальном видео: {real_video_error:.6f}")
        logger.info(f"Отношение (шум/реальное): {avg_noise_error / real_video_error:.2f}x")
        
        if avg_noise_error > detector.threshold:
            logger.info(f"✅ Модель правильно различает шум и реальные данные")
        else:
            logger.warning(f"⚠️ Модель может путать шум с нормальными данными")
    
    return results


if __name__ == "__main__":
    test_random_noise()

