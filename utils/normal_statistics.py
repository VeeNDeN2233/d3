"""
Вычисление нормальных статистик движений из тренировочных данных MINI-RGBD.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from utils.data_loader import MiniRGBDDataLoader
from utils.pose_processor import PoseProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Кэш для нормальных статистик
_normal_stats_cache: Optional[Dict] = None


def calculate_normal_statistics(
    config_path: str = "config.yaml",
    force_recalculate: bool = False,
) -> Dict:
    """
    Вычислить нормальные статистики движений из тренировочных данных MINI-RGBD.
    
    Args:
        config_path: Путь к конфигурации
        force_recalculate: Принудительно пересчитать даже если есть кэш
    
    Returns:
        Словарь с нормальными статистиками:
        - joint_amplitudes: средние амплитуды для каждого сустава (25,)
        - joint_velocities: средние скорости для каждого сустава (25,)
        - left_right_ratios: нормальные соотношения левая/правая для рук и ног
        - overall_statistics: общие статистики
    """
    global _normal_stats_cache
    
    # Проверяем кэш
    cache_path = Path("checkpoints/normal_statistics.json")
    if not force_recalculate and _normal_stats_cache is not None:
        return _normal_stats_cache
    
    if not force_recalculate and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                _normal_stats_cache = json.load(f)
                logger.info(f"Нормальные статистики загружены из кэша: {cache_path}")
                return _normal_stats_cache
        except Exception as e:
            logger.warning(f"Ошибка загрузки кэша: {e}, пересчитываем...")
    
    logger.info("Вычисление нормальных статистик из тренировочных данных MINI-RGBD...")
    
    # Загружаем конфигурацию
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Загружаем тренировочные данные
    data_loader = MiniRGBDDataLoader(
        data_root=config["data"]["mini_rgbd_path"],
        train_sequences=config["data"]["train_sequences"],
        val_sequences=config["data"]["val_sequences"],
        test_sequences=config["data"]["test_sequences"],
        model_complexity=config["pose"]["model_complexity"],
        min_detection_confidence=config["pose"]["min_detection_confidence"],
        min_tracking_confidence=config["pose"]["min_tracking_confidence"],
    )
    
    pose_processor = PoseProcessor(
        sequence_length=config["pose"]["sequence_length"],
        sequence_stride=config["pose"]["sequence_stride"],
        normalize=config["pose"]["normalize"],
        normalize_relative_to=config["pose"]["normalize_relative_to"],
        target_hip_distance=config["pose"].get("target_hip_distance"),
        normalize_by_body=config["pose"].get("normalize_by_body", False),
        rotate_to_canonical=config["pose"].get("rotate_to_canonical", False),
    )
    
    # Загружаем тренировочные данные
    train_images, train_keypoints = data_loader.load_train_data(
        max_frames_per_seq=config["data"]["max_frames_per_seq"]
    )
    
    # Обрабатываем ключевые точки
    sequences = pose_processor.process_keypoints(train_keypoints)
    
    if len(sequences) == 0:
        raise RuntimeError("Нет валидных последовательностей в тренировочных данных!")
    
    # Преобразуем в массив (N, seq_len, 75) = (N, 30, 25*3)
    sequences_array = np.array([pose_processor.flatten_sequence(seq) for seq in sequences])
    n_seqs, seq_len, _ = sequences_array.shape
    sequences_reshaped = sequences_array.reshape(n_seqs, seq_len, 25, 3)
    
    logger.info(f"Обработано {n_seqs} нормальных последовательностей из тренировочных данных")
    
    # Вычисляем амплитуды движений (стандартное отклонение по времени для каждого сустава)
    amplitudes = []
    for seq in sequences_reshaped:
        amp = np.std(seq, axis=0)  # (25, 3)
        amplitudes.append(np.linalg.norm(amp, axis=1))  # (25,)
    amplitudes = np.array(amplitudes)  # (N, 25)
    
    # Средние амплитуды для каждого сустава
    joint_amplitudes = np.mean(amplitudes, axis=0)  # (25,)
    joint_amplitudes_std = np.std(amplitudes, axis=0)  # (25,)
    
    # Вычисляем скорости движений (изменение позиции между кадрами)
    velocities = []
    for seq in sequences_reshaped:
        diffs = np.diff(seq, axis=0)  # (seq_len-1, 25, 3)
        vel = np.linalg.norm(diffs, axis=2)  # (seq_len-1, 25)
        velocities.append(np.mean(vel, axis=0))  # (25,)
    velocities = np.array(velocities)  # (N, 25)
    
    # Средние скорости для каждого сустава
    joint_velocities = np.mean(velocities, axis=0)  # (25,)
    joint_velocities_std = np.std(velocities, axis=0)  # (25,)
    
    # Нормальные соотношения левая/правая
    from utils.anomaly_analyzer import LEFT_JOINTS, RIGHT_JOINTS
    
    # Для рук
    left_arm_amplitudes = amplitudes[:, LEFT_JOINTS["arm"]].mean(axis=1)  # (N,)
    right_arm_amplitudes = amplitudes[:, RIGHT_JOINTS["arm"]].mean(axis=1)  # (N,)
    arm_ratios = left_arm_amplitudes / (right_arm_amplitudes + 1e-6)  # (N,)
    normal_arm_ratio_mean = np.mean(arm_ratios)
    normal_arm_ratio_std = np.std(arm_ratios)
    
    # Для ног
    left_leg_amplitudes = amplitudes[:, LEFT_JOINTS["leg"]].mean(axis=1)  # (N,)
    right_leg_amplitudes = amplitudes[:, RIGHT_JOINTS["leg"]].mean(axis=1)  # (N,)
    leg_ratios = left_leg_amplitudes / (right_leg_amplitudes + 1e-6)  # (N,)
    normal_leg_ratio_mean = np.mean(leg_ratios)
    normal_leg_ratio_std = np.std(leg_ratios)
    
    # Общие статистики
    overall_amplitude_mean = np.mean(amplitudes)
    overall_amplitude_std = np.std(amplitudes)
    overall_velocity_mean = np.mean(velocities)
    overall_velocity_std = np.std(velocities)
    
    stats = {
        "joint_amplitudes": joint_amplitudes.tolist(),
        "joint_amplitudes_std": joint_amplitudes_std.tolist(),
        "joint_velocities": joint_velocities.tolist(),
        "joint_velocities_std": joint_velocities_std.tolist(),
        "left_right_ratios": {
            "arm": {
                "mean": float(normal_arm_ratio_mean),
                "std": float(normal_arm_ratio_std),
            },
            "leg": {
                "mean": float(normal_leg_ratio_mean),
                "std": float(normal_leg_ratio_std),
            },
        },
        "overall_statistics": {
            "amplitude_mean": float(overall_amplitude_mean),
            "amplitude_std": float(overall_amplitude_std),
            "velocity_mean": float(overall_velocity_mean),
            "velocity_std": float(overall_velocity_std),
        },
        "sample_size": int(n_seqs),
        "source": "MINI-RGBD training data",
    }
    
    # Сохраняем в кэш
    _normal_stats_cache = stats
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Нормальные статистики сохранены: {cache_path}")
    logger.info(f"  Образцов: {n_seqs}")
    logger.info(f"  Средняя амплитуда: {overall_amplitude_mean:.6f} ± {overall_amplitude_std:.6f}")
    logger.info(f"  Средняя скорость: {overall_velocity_mean:.6f} ± {overall_velocity_std:.6f}")
    
    return stats


def get_normal_statistics(config_path: str = "config.yaml") -> Dict:
    """Получить нормальные статистики (с кэшированием)."""
    return calculate_normal_statistics(config_path, force_recalculate=False)

