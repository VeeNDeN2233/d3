"""
Генератор синтетических аномалий для тестирования системы детекции.

Создает различные типы аномалий движений младенцев:
- Асимметрия движений
- Тремор конечностей
- Ригидность (скованность)
"""

import logging
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticAnomalyGenerator:
    """
    Генератор синтетических аномалий движений.
    
    Принимает нормальную последовательность и создает различные типы аномалий.
    """

    def __init__(self, normal_sequence: np.ndarray):
        """
        Инициализация генератора.
        
        Args:
            normal_sequence: Нормальная последовательность формы [T, 75] или [T, 25, 3]
        """
        self.normal = normal_sequence.copy()
        
        # Определяем формат данных
        if len(self.normal.shape) == 2:
            # [T, 75] - плоский формат
            self.flat_format = True
            self.T = self.normal.shape[0]
            # Преобразуем в [T, 25, 3] для работы
            self.normal_3d = self.normal.reshape(self.T, 25, 3)
        else:
            # [T, 25, 3] - 3D формат
            self.flat_format = False
            self.T = self.normal.shape[0]
            self.normal_3d = self.normal
        
        logger.info(f"Инициализирован генератор аномалий: T={self.T}")

    def _to_output_format(self, sequence: np.ndarray) -> np.ndarray:
        """Преобразовать в исходный формат."""
        if self.flat_format:
            return sequence.reshape(self.T, 75)
        return sequence

    def create_asymmetry(
        self, strength: float = 0.3, side: str = "left"
    ) -> np.ndarray:
        """
        Создать асимметрию движений (левая сторона vs правая).
        
        Args:
            strength: Сила аномалии (0.0 - нет, 1.0 - максимальная)
            side: Какая сторона ослаблена ("left" или "right")
        
        Returns:
            Аномальная последовательность
        """
        anomalous = self.normal_3d.copy()
        
        # Индексы суставов MINI-RGBD (25 суставов)
        # Левая сторона: 1, 4, 7, 10, 13, 16, 19, 22 (leftThigh, leftCalf, leftFoot, leftToes, leftShoulder, leftUpperArm, leftForeArm, leftHand, leftFingers)
        # Правая сторона: 2, 5, 8, 11, 14, 17, 20, 23 (rightThigh, rightCalf, rightFoot, rightToes, rightShoulder, rightUpperArm, rightForeArm, rightHand, rightFingers)
        
        if side == "left":
            left_joints = [1, 4, 7, 10, 13, 16, 19, 20, 22]  # левые суставы
            # Ослабляем движения левой стороны
            for joint_idx in left_joints:
                center = np.mean(anomalous[:, joint_idx, :], axis=0, keepdims=True)
                anomalous[:, joint_idx, :] = center + (anomalous[:, joint_idx, :] - center) * (1 - strength)
        else:
            right_joints = [2, 5, 8, 11, 14, 17, 19, 21, 23]  # правые суставы
            # Ослабляем движения правой стороны
            for joint_idx in right_joints:
                center = np.mean(anomalous[:, joint_idx, :], axis=0, keepdims=True)
                anomalous[:, joint_idx, :] = center + (anomalous[:, joint_idx, :] - center) * (1 - strength)
        
        logger.info(f"Создана асимметрия: side={side}, strength={strength}")
        return self._to_output_format(anomalous)

    def create_tremor(
        self, frequency: float = 10.0, amplitude: float = 0.1
    ) -> np.ndarray:
        """
        Создать тремор (мелкая дрожь в конечностях).
        
        Args:
            frequency: Частота тремора (колебаний в секунду)
            amplitude: Амплитуда тремора
        
        Returns:
            Аномальная последовательность с тремором
        """
        anomalous = self.normal_3d.copy()
        t = np.arange(self.T)
        
        # Добавляем синусоидальный тремор к конечностям
        # Руки и ноги: 7, 8, 10, 11, 20, 21, 22, 23 (feet, toes, hands, fingers)
        limb_joints = [7, 8, 10, 11, 20, 21, 22, 23]
        
        for joint_idx in limb_joints:
            # Тремор в разных направлениях
            for dim in range(3):
                tremor = amplitude * np.sin(2 * np.pi * frequency * t / self.T)
                # Добавляем случайную фазу для реалистичности
                phase = np.random.uniform(0, 2 * np.pi)
                tremor = amplitude * np.sin(2 * np.pi * frequency * t / self.T + phase)
                anomalous[:, joint_idx, dim] += tremor
        
        logger.info(f"Создан тремор: frequency={frequency}, amplitude={amplitude}")
        return self._to_output_format(anomalous)

    def create_rigidity(self, stiffness: float = 0.5) -> np.ndarray:
        """
        Создать ригидность (скованность движений).
        
        Args:
            stiffness: Уровень скованности (0.0 - нет, 1.0 - полная неподвижность)
        
        Returns:
            Аномальная последовательность с ригидностью
        """
        anomalous = self.normal_3d.copy()
        
        # Вычисляем центр (среднее положение)
        center = np.mean(self.normal_3d, axis=0, keepdims=True)
        
        # Уменьшаем амплитуду всех движений
        anomalous = center + (anomalous - center) * (1 - stiffness)
        
        logger.info(f"Создана ригидность: stiffness={stiffness}")
        return self._to_output_format(anomalous)

    def create_hyperactivity(
        self, intensity: float = 0.3, randomness: float = 0.1
    ) -> np.ndarray:
        """
        Создать гиперкинез (избыточная активность движений).
        
        Args:
            intensity: Интенсивность гиперкинеза
            randomness: Случайность движений
        
        Returns:
            Аномальная последовательность с гиперкинезом
        """
        anomalous = self.normal_3d.copy()
        
        # Увеличиваем амплитуду движений
        center = np.mean(self.normal_3d, axis=0, keepdims=True)
        anomalous = center + (anomalous - center) * (1 + intensity)
        
        # Добавляем случайные движения
        if randomness > 0:
            noise = np.random.normal(0, randomness, anomalous.shape)
            anomalous += noise
        
        logger.info(f"Создан гиперкинез: intensity={intensity}, randomness={randomness}")
        return self._to_output_format(anomalous)

    def create_limited_range(
        self, joint_indices: list, reduction: float = 0.5
    ) -> np.ndarray:
        """
        Создать ограниченный диапазон движений в определенных суставах.
        
        Args:
            joint_indices: Индексы суставов с ограничением
            reduction: Степень ограничения (0.0 - нет, 1.0 - полное)
        
        Returns:
            Аномальная последовательность
        """
        anomalous = self.normal_3d.copy()
        
        for joint_idx in joint_indices:
            if 0 <= joint_idx < 25:
                # Ограничиваем диапазон движений
                center = np.mean(anomalous[:, joint_idx, :], axis=0, keepdims=True)
                anomalous[:, joint_idx, :] = center + (anomalous[:, joint_idx, :] - center) * (1 - reduction)
        
        logger.info(f"Создано ограничение диапазона: joints={joint_indices}, reduction={reduction}")
        return self._to_output_format(anomalous)

    def create_combined(
        self,
        asymmetry_strength: Optional[float] = None,
        tremor_amplitude: Optional[float] = None,
        rigidity_stiffness: Optional[float] = None,
    ) -> np.ndarray:
        """
        Создать комбинированную аномалию.
        
        Args:
            asymmetry_strength: Сила асимметрии (None = не применять)
            tremor_amplitude: Амплитуда тремора (None = не применять)
            rigidity_stiffness: Скованность (None = не применять)
        
        Returns:
            Аномальная последовательность
        """
        anomalous = self.normal_3d.copy()
        
        if asymmetry_strength is not None:
            temp = self.create_asymmetry(asymmetry_strength)
            if not self.flat_format:
                temp = temp.reshape(self.T, 25, 3)
            anomalous = anomalous * 0.5 + temp.reshape(self.T, 25, 3) * 0.5
        
        if tremor_amplitude is not None:
            temp = self.create_tremor(amplitude=tremor_amplitude)
            if not self.flat_format:
                temp = temp.reshape(self.T, 25, 3)
            anomalous += (temp.reshape(self.T, 25, 3) - anomalous) * 0.3
        
        if rigidity_stiffness is not None:
            temp = self.create_rigidity(rigidity_stiffness)
            if not self.flat_format:
                temp = temp.reshape(self.T, 25, 3)
            anomalous = anomalous * 0.7 + temp.reshape(self.T, 25, 3) * 0.3
        
        logger.info("Создана комбинированная аномалия")
        return self._to_output_format(anomalous)


def generate_anomalies_from_sequences(
    normal_sequences: list, output_dir: str = "test_videos/anomalies"
) -> dict:
    """
    Генерировать аномалии из набора нормальных последовательностей.
    
    Args:
        normal_sequences: Список нормальных последовательностей
        output_dir: Директория для сохранения
    
    Returns:
        Словарь с аномальными последовательностями
    """
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_anomalies = {}
    
    for i, normal_seq in enumerate(normal_sequences):
        generator = SyntheticAnomalyGenerator(normal_seq)
        
        anomalies = {
            "asymmetry_left": generator.create_asymmetry(strength=0.3, side="left"),
            "asymmetry_right": generator.create_asymmetry(strength=0.3, side="right"),
            "tremor": generator.create_tremor(frequency=10.0, amplitude=0.1),
            "rigidity": generator.create_rigidity(stiffness=0.5),
            "hyperactivity": generator.create_hyperactivity(intensity=0.3),
        }
        
        all_anomalies[f"sequence_{i}"] = anomalies
        
        # Сохраняем в файлы
        for anomaly_type, anomaly_seq in anomalies.items():
            filename = output_path / f"seq_{i}_{anomaly_type}.npy"
            np.save(filename, anomaly_seq)
    
    logger.info(f"Сгенерировано {len(all_anomalies)} наборов аномалий")
    return all_anomalies

