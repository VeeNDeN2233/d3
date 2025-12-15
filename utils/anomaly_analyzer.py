"""
Детальный анализ аномалий движений по суставам и конечностям.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

# Названия суставов MINI-RGBD (25 суставов)
JOINT_NAMES = [
    "global", "leftThigh", "rightThigh", "spine", "leftCalf", "rightCalf",
    "spine1", "leftFoot", "rightFoot", "spine2", "leftToes", "rightToes",
    "neck", "leftShoulder", "rightShoulder", "head",
    "leftUpperArm", "rightUpperArm", "leftForeArm", "rightForeArm",
    "leftHand", "rightHand", "leftFingers", "rightFingers", "noseVertex"
]

# Индексы левых и правых суставов
LEFT_JOINTS = {
    "arm": [13, 16, 18, 20, 22],  # leftShoulder, leftUpperArm, leftForeArm, leftHand, leftFingers
    "leg": [1, 4, 7, 10],  # leftThigh, leftCalf, leftFoot, leftToes
}
RIGHT_JOINTS = {
    "arm": [14, 17, 19, 21, 23],  # rightShoulder, rightUpperArm, rightForeArm, rightHand, rightFingers
    "leg": [2, 5, 8, 11],  # rightThigh, rightCalf, rightFoot, rightToes
}

logger = logging.getLogger(__name__)


def analyze_joint_errors(
    sequences: np.ndarray,
    reconstruction_errors: np.ndarray,
    threshold: float,
) -> Dict:
    """
    Анализ ошибок реконструкции по суставам.
    
    Args:
        sequences: Последовательности ключевых точек (N, seq_len, 75)
                   где 75 = 25 суставов × 3 координаты (x, y, z)
        reconstruction_errors: Ошибки реконструкции для каждой последовательности (N,)
        threshold: Порог аномалии
    
    Returns:
        Словарь с детальным анализом
    """
    if len(sequences) == 0:
        return {}
    
    # Определяем аномальные последовательности
    anomalous_mask = reconstruction_errors > threshold
    normal_mask = ~anomalous_mask
    
    if np.sum(anomalous_mask) == 0:
        return {
            "has_anomalies": False,
            "message": "Аномалий не обнаружено"
        }
    
    # Анализируем аномальные последовательности
    anomalous_sequences = sequences[anomalous_mask]
    normal_sequences = sequences[normal_mask] if np.sum(normal_mask) > 0 else sequences
    
    # Вычисляем статистики по суставам для нормальных и аномальных последовательностей
    # sequences shape: (N, seq_len, 75) = (N, 30, 25*3)
    # Нужно переформатировать в (N, seq_len, 25, 3)
    n_seqs, seq_len, _ = sequences.shape
    sequences_reshaped = sequences.reshape(n_seqs, seq_len, 25, 3)
    
    anomalous_reshaped = sequences_reshaped[anomalous_mask]
    normal_reshaped = sequences_reshaped[normal_mask] if np.sum(normal_mask) > 0 else sequences_reshaped
    
    # Вычисляем амплитуду движений для каждого сустава (стандартное отклонение по времени)
    # Для нормальных последовательностей
    normal_amplitudes = np.std(normal_reshaped, axis=1)  # (N_normal, 25, 3)
    normal_amplitude_mean = np.mean(normal_amplitudes, axis=0)  # (25, 3)
    normal_amplitude_total = np.linalg.norm(normal_amplitude_mean, axis=1)  # (25,)
    
    # Для аномальных последовательностей
    anomalous_amplitudes = np.std(anomalous_reshaped, axis=1)  # (N_anomalous, 25, 3)
    anomalous_amplitude_mean = np.mean(anomalous_amplitudes, axis=0)  # (25, 3)
    anomalous_amplitude_total = np.linalg.norm(anomalous_amplitude_mean, axis=1)  # (25,)
    
    # Вычисляем скорость движений (изменение позиции между кадрами)
    # Для нормальных
    normal_velocities = []
    for seq in normal_reshaped:
        # Вычисляем изменения между соседними кадрами
        diffs = np.diff(seq, axis=0)  # (seq_len-1, 25, 3)
        velocities = np.linalg.norm(diffs, axis=2)  # (seq_len-1, 25)
        normal_velocities.append(np.mean(velocities, axis=0))  # (25,)
    normal_velocity_mean = np.mean(normal_velocities, axis=0) if normal_velocities else np.zeros(25)
    
    # Для аномальных
    anomalous_velocities = []
    for seq in anomalous_reshaped:
        diffs = np.diff(seq, axis=0)
        velocities = np.linalg.norm(diffs, axis=2)
        anomalous_velocities.append(np.mean(velocities, axis=0))
    anomalous_velocity_mean = np.mean(anomalous_velocities, axis=0) if anomalous_velocities else np.zeros(25)
    
    # Анализ асимметрии
    asymmetry_analysis = analyze_asymmetry(
        anomalous_reshaped, normal_reshaped,
        anomalous_amplitude_total, normal_amplitude_total,
        anomalous_velocity_mean, normal_velocity_mean
    )
    
    # Анализ конкретных суставов с высокой ошибкой
    joint_analysis = analyze_specific_joints(
        anomalous_amplitude_total, normal_amplitude_total,
        anomalous_velocity_mean, normal_velocity_mean
    )
    
    # Анализ скорости движений
    speed_analysis = analyze_movement_speed(
        anomalous_velocity_mean, normal_velocity_mean
    )
    
    # Анализ амплитуды движений
    amplitude_analysis = analyze_movement_amplitude(
        anomalous_amplitude_total, normal_amplitude_total
    )
    
    return {
        "has_anomalies": True,
        "anomalous_sequences_count": int(np.sum(anomalous_mask)),
        "total_sequences": len(sequences),
        "asymmetry": asymmetry_analysis,
        "joint_analysis": joint_analysis,
        "speed_analysis": speed_analysis,
        "amplitude_analysis": amplitude_analysis,
    }


def analyze_asymmetry(
    anomalous_reshaped: np.ndarray,
    normal_reshaped: np.ndarray,
    anomalous_amplitude: np.ndarray,
    normal_amplitude: np.ndarray,
    anomalous_velocity: np.ndarray,
    normal_velocity: np.ndarray,
) -> Dict:
    """Анализ асимметрии движений между левой и правой сторонами."""
    findings = []
    
    # Асимметрия амплитуды для рук
    left_arm_amplitude = np.mean(anomalous_amplitude[LEFT_JOINTS["arm"]])
    right_arm_amplitude = np.mean(anomalous_amplitude[RIGHT_JOINTS["arm"]])
    left_arm_normal = np.mean(normal_amplitude[LEFT_JOINTS["arm"]])
    right_arm_normal = np.mean(normal_amplitude[RIGHT_JOINTS["arm"]])
    
    arm_asymmetry_ratio = abs(left_arm_amplitude - right_arm_amplitude) / max(left_arm_amplitude, right_arm_amplitude, 1e-6)
    arm_normal_ratio = abs(left_arm_normal - right_arm_normal) / max(left_arm_normal, right_arm_normal, 1e-6)
    
    if arm_asymmetry_ratio > 0.3 and arm_asymmetry_ratio > arm_normal_ratio * 1.5:
        if left_arm_amplitude < right_arm_amplitude * 0.7:
            findings.append({
                "type": "asymmetry",
                "body_part": "руки",
                "description": "Сниженная амплитуда движений левой руки по сравнению с правой",
                "severity": "medium" if arm_asymmetry_ratio < 0.5 else "high",
                "data": {
                    "left_amplitude": float(left_arm_amplitude),
                    "right_amplitude": float(right_arm_amplitude),
                    "asymmetry_ratio": float(arm_asymmetry_ratio),
                }
            })
        elif right_arm_amplitude < left_arm_amplitude * 0.7:
            findings.append({
                "type": "asymmetry",
                "body_part": "руки",
                "description": "Сниженная амплитуда движений правой руки по сравнению с левой",
                "severity": "medium" if arm_asymmetry_ratio < 0.5 else "high",
                "data": {
                    "left_amplitude": float(left_arm_amplitude),
                    "right_amplitude": float(right_arm_amplitude),
                    "asymmetry_ratio": float(arm_asymmetry_ratio),
                }
            })
    
    # Асимметрия амплитуды для ног
    left_leg_amplitude = np.mean(anomalous_amplitude[LEFT_JOINTS["leg"]])
    right_leg_amplitude = np.mean(anomalous_amplitude[RIGHT_JOINTS["leg"]])
    left_leg_normal = np.mean(normal_amplitude[LEFT_JOINTS["leg"]])
    right_leg_normal = np.mean(normal_amplitude[RIGHT_JOINTS["leg"]])
    
    leg_asymmetry_ratio = abs(left_leg_amplitude - right_leg_amplitude) / max(left_leg_amplitude, right_leg_amplitude, 1e-6)
    leg_normal_ratio = abs(left_leg_normal - right_leg_normal) / max(left_leg_normal, right_leg_normal, 1e-6)
    
    if leg_asymmetry_ratio > 0.3 and leg_asymmetry_ratio > leg_normal_ratio * 1.5:
        if left_leg_amplitude < right_leg_amplitude * 0.7:
            findings.append({
                "type": "asymmetry",
                "body_part": "ноги",
                "description": "Сниженная амплитуда движений левой ноги по сравнению с правой",
                "severity": "medium" if leg_asymmetry_ratio < 0.5 else "high",
                "data": {
                    "left_amplitude": float(left_leg_amplitude),
                    "right_amplitude": float(right_leg_amplitude),
                    "asymmetry_ratio": float(leg_asymmetry_ratio),
                }
            })
        elif right_leg_amplitude < left_leg_amplitude * 0.7:
            findings.append({
                "type": "asymmetry",
                "body_part": "ноги",
                "description": "Сниженная амплитуда движений правой ноги по сравнению с левой",
                "severity": "medium" if leg_asymmetry_ratio < 0.5 else "high",
                "data": {
                    "left_amplitude": float(left_leg_amplitude),
                    "right_amplitude": float(right_leg_amplitude),
                    "asymmetry_ratio": float(leg_asymmetry_ratio),
                }
            })
    
    return {
        "findings": findings,
        "has_asymmetry": len(findings) > 0
    }


def analyze_specific_joints(
    anomalous_amplitude: np.ndarray,
    normal_amplitude: np.ndarray,
    anomalous_velocity: np.ndarray,
    normal_velocity: np.ndarray,
) -> Dict:
    """Анализ конкретных суставов с отклонениями."""
    findings = []
    
    # Находим суставы с аномально низкой амплитудой (отсутствие движения)
    amplitude_ratio = anomalous_amplitude / (normal_amplitude + 1e-6)
    low_amplitude_joints = np.where(amplitude_ratio < 0.5)[0]
    
    for joint_idx in low_amplitude_joints:
        if joint_idx < len(JOINT_NAMES):
            joint_name = JOINT_NAMES[joint_idx]
            # Пропускаем центральные суставы (spine, neck, head)
            if joint_name not in ["global", "spine", "spine1", "spine2", "neck", "head", "noseVertex"]:
                findings.append({
                    "type": "reduced_movement",
                    "joint": joint_name,
                    "description": f"Сниженная амплитуда движений {joint_name}",
                    "severity": "medium",
                    "data": {
                        "anomalous_amplitude": float(anomalous_amplitude[joint_idx]),
                        "normal_amplitude": float(normal_amplitude[joint_idx]),
                        "ratio": float(amplitude_ratio[joint_idx]),
                    }
                })
    
    # Находим суставы с аномально высокой скоростью
    velocity_ratio = anomalous_velocity / (normal_velocity + 1e-6)
    high_velocity_joints = np.where(velocity_ratio > 1.5)[0]
    
    for joint_idx in high_velocity_joints:
        if joint_idx < len(JOINT_NAMES):
            joint_name = JOINT_NAMES[joint_idx]
            if joint_name not in ["global", "spine", "spine1", "spine2", "neck", "head", "noseVertex"]:
                findings.append({
                    "type": "high_speed",
                    "joint": joint_name,
                    "description": f"Повышенная скорость движений {joint_name}",
                    "severity": "medium",
                    "data": {
                        "anomalous_velocity": float(anomalous_velocity[joint_idx]),
                        "normal_velocity": float(normal_velocity[joint_idx]),
                        "ratio": float(velocity_ratio[joint_idx]),
                    }
                })
    
    return {
        "findings": findings,
        "affected_joints": [JOINT_NAMES[i] for i in low_amplitude_joints if i < len(JOINT_NAMES)]
    }


def analyze_movement_speed(
    anomalous_velocity: np.ndarray,
    normal_velocity: np.ndarray,
) -> Dict:
    """Анализ скорости движений."""
    findings = []
    
    # Общая скорость
    total_anomalous_velocity = np.mean(anomalous_velocity)
    total_normal_velocity = np.mean(normal_velocity)
    
    if total_anomalous_velocity > total_normal_velocity * 1.5:
        findings.append({
            "type": "overall_high_speed",
            "description": "Общая скорость движений выше нормы",
            "severity": "medium",
            "data": {
                "anomalous_speed": float(total_anomalous_velocity),
                "normal_speed": float(total_normal_velocity),
                "ratio": float(total_anomalous_velocity / (total_normal_velocity + 1e-6)),
            }
        })
    elif total_anomalous_velocity < total_normal_velocity * 0.5:
        findings.append({
            "type": "overall_low_speed",
            "description": "Общая скорость движений ниже нормы",
            "severity": "medium",
            "data": {
                "anomalous_speed": float(total_anomalous_velocity),
                "normal_speed": float(total_normal_velocity),
                "ratio": float(total_anomalous_velocity / (total_normal_velocity + 1e-6)),
            }
        })
    
    # Скорость рук
    left_arm_velocity = np.mean(anomalous_velocity[LEFT_JOINTS["arm"]])
    right_arm_velocity = np.mean(anomalous_velocity[RIGHT_JOINTS["arm"]])
    left_arm_normal = np.mean(normal_velocity[LEFT_JOINTS["arm"]])
    right_arm_normal = np.mean(normal_velocity[RIGHT_JOINTS["arm"]])
    
    if left_arm_velocity > left_arm_normal * 1.5:
        findings.append({
            "type": "high_speed",
            "body_part": "левая рука",
            "description": "Повышенная скорость движений левой руки",
            "severity": "medium",
        })
    if right_arm_velocity > right_arm_normal * 1.5:
        findings.append({
            "type": "high_speed",
            "body_part": "правая рука",
            "description": "Повышенная скорость движений правой руки",
            "severity": "medium",
        })
    
    return {
        "findings": findings,
        "has_speed_anomalies": len(findings) > 0
    }


def analyze_movement_amplitude(
    anomalous_amplitude: np.ndarray,
    normal_amplitude: np.ndarray,
) -> Dict:
    """Анализ амплитуды движений."""
    findings = []
    
    # Общая амплитуда
    total_anomalous_amplitude = np.mean(anomalous_amplitude)
    total_normal_amplitude = np.mean(normal_amplitude)
    
    if total_anomalous_amplitude < total_normal_amplitude * 0.5:
        findings.append({
            "type": "overall_reduced_amplitude",
            "description": "Общая амплитуда движений снижена",
            "severity": "medium",
            "data": {
                "anomalous_amplitude": float(total_anomalous_amplitude),
                "normal_amplitude": float(total_normal_amplitude),
                "ratio": float(total_anomalous_amplitude / (total_normal_amplitude + 1e-6)),
            }
        })
    
    return {
        "findings": findings,
        "has_amplitude_anomalies": len(findings) > 0
    }

