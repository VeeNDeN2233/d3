"""
Анализ активности движений в видео.
"""

import json
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_movement_activity(video_path: Path) -> dict:
    """
    Проанализировать активность движений в видео.
    
    Метрики:
    1. Скорость изменения ключевых точек (velocity)
    2. Амплитуда движений (amplitude)
    3. Частота движений (frequency)
    """
    logger.info(f"Анализ активности движений: {video_path}")
    
    # Загружаем конфигурацию
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Инициализация процессора
    video_processor = VideoProcessor(
        model_complexity=config["pose"]["model_complexity"],
        min_detection_confidence=config["pose"]["min_detection_confidence"],
        min_tracking_confidence=config["pose"]["min_tracking_confidence"],
    )
    
    # Обрабатываем видео
    temp_output = video_path.parent / f"temp_{video_path.name}"
    result = video_processor._process_video_sync(
        str(video_path), str(temp_output), save_keypoints=True
    )
    
    if not result["success"]:
        raise RuntimeError(f"Ошибка обработки видео: {result.get('error')}")
    
    # Загружаем ключевые точки
    keypoints_path = Path(result["keypoints_path"]) / "keypoints.json"
    with open(keypoints_path, "r", encoding="utf-8") as f:
        keypoints_data = json.load(f)
    
    # Извлекаем ключевые точки
    keypoints_list = []
    for frame_data in keypoints_data.get("frames", []):
        landmarks = frame_data.get("landmarks")
        if landmarks:
            kp = np.array(
                [[lm["x"], lm["y"], lm["z"]] for lm in landmarks],
                dtype=np.float32,
            )
            keypoints_list.append(kp)
        else:
            keypoints_list.append(None)
    
    # Фильтруем None
    valid_keypoints = [kp for kp in keypoints_list if kp is not None]
    
    if len(valid_keypoints) < 2:
        logger.error("Недостаточно кадров для анализа")
        return {}
    
    # Вычисляем метрики движения
    velocities = []
    amplitudes = []
    
    for i in range(1, len(valid_keypoints)):
        prev_kp = valid_keypoints[i-1]
        curr_kp = valid_keypoints[i]
        
        # Скорость изменения (velocity) - евклидово расстояние между кадрами
        diff = curr_kp - prev_kp
        velocity = np.linalg.norm(diff, axis=1).mean()  # средняя скорость по всем точкам
        velocities.append(velocity)
        
        # Амплитуда движений - максимальное смещение точки
        max_displacement = np.linalg.norm(diff, axis=1).max()
        amplitudes.append(max_displacement)
    
    velocities = np.array(velocities)
    amplitudes = np.array(amplitudes)
    
    # Статистики
    stats = {
        "total_frames": len(keypoints_list),
        "valid_frames": len(valid_keypoints),
        "detection_rate": len(valid_keypoints) / len(keypoints_list) * 100,
        "movement_velocity": {
            "mean": float(velocities.mean()),
            "std": float(velocities.std()),
            "min": float(velocities.min()),
            "max": float(velocities.max()),
        },
        "movement_amplitude": {
            "mean": float(amplitudes.mean()),
            "std": float(amplitudes.std()),
            "min": float(amplitudes.min()),
            "max": float(amplitudes.max()),
        },
        "activity_level": "unknown",
    }
    
    # Определяем уровень активности
    mean_velocity = stats["movement_velocity"]["mean"]
    if mean_velocity > 0.01:
        stats["activity_level"] = "high"
    elif mean_velocity > 0.005:
        stats["activity_level"] = "medium"
    else:
        stats["activity_level"] = "low"
    
    logger.info(f"\nАктивность движений:")
    logger.info(f"  Уровень: {stats['activity_level']}")
    logger.info(f"  Средняя скорость: {mean_velocity:.6f}")
    logger.info(f"  Средняя амплитуда: {stats['movement_amplitude']['mean']:.6f}")
    logger.info(f"  Detection rate: {stats['detection_rate']:.1f}%")
    
    # Удаляем временный файл
    if temp_output.exists():
        temp_output.unlink()
    
    return stats


if __name__ == "__main__":
    video_path = Path("test_videos/baby.mp4")
    if video_path.exists():
        stats = analyze_movement_activity(video_path)
        
        # Сохраняем результаты
        output_path = Path("results/baby_advanced_model/activity_analysis.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nРезультаты сохранены: {output_path}")
    else:
        logger.error(f"Видео не найдено: {video_path}")

