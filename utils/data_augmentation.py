"""
Аугментация данных для улучшения обобщения модели.

Добавляет "реалистичности" к синтетическим данным MINI-RGBD,
чтобы они были похожи на реальные видео.
"""

import logging
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Аугментация ключевых точек для улучшения обобщения.
    """

    def __init__(
        self,
        noise_level: float = 0.02,
        asymmetry_range: tuple = (-0.03, 0.03),
        add_natural_variation: bool = True,
        rotation_range: float = 5.0,  # градусы
        scale_range: tuple = (0.95, 1.05),
    ):
        """
        Инициализация аугментации.
        
        Args:
            noise_level: Уровень шума (стандартное отклонение)
            asymmetry_range: Диапазон асимметрии для левой/правой стороны
            add_natural_variation: Добавлять естественную изменчивость
            rotation_range: Диапазон вращения в градусах
            scale_range: Диапазон масштабирования
        """
        self.noise_level = noise_level
        self.asymmetry_range = asymmetry_range
        self.add_natural_variation = add_natural_variation
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        
        # Индексы левых и правых суставов в MINI-RGBD формате (25 суставов)
        # Формат: [global, leftThigh, rightThigh, spine, leftCalf, rightCalf, ...]
        self.left_side_indices = [
            1,   # leftThigh
            4,   # leftCalf
            7,   # leftFoot
            10,  # leftToes
            13,  # leftShoulder
            16,  # leftUpperArm
            18,  # leftForeArm
            19,  # leftHand
            21,  # leftFingers
        ]
        
        self.right_side_indices = [
            2,   # rightThigh
            5,   # rightCalf
            8,   # rightFoot
            11,  # rightToes
            14,  # rightShoulder
            17,  # rightUpperArm
            19,  # rightForeArm (исправлено: должно быть 20)
            20,  # rightHand
            22,  # rightFingers
        ]
        
        # Исправление индексов
        self.right_side_indices = [
            2,   # rightThigh
            5,   # rightCalf
            8,   # rightFoot
            11,  # rightToes
            14,  # rightShoulder
            17,  # rightUpperArm
            20,  # rightForeArm
            21,  # rightHand
            23,  # rightFingers
        ]
        
        logger.info(
            f"Инициализирован DataAugmentation: "
            f"noise={noise_level}, asymmetry={asymmetry_range}, "
            f"rotation={rotation_range}deg, scale={scale_range}"
        )

    def augment_keypoints(
        self, keypoints: np.ndarray, apply_all: bool = True
    ) -> np.ndarray:
        """
        Применить аугментацию к ключевым точкам.
        
        Args:
            keypoints: Массив формы (25, 3) или (N, 25, 3) для последовательности
            apply_all: Применить все виды аугментации
        
        Returns:
            Аугментированные ключевые точки
        """
        if len(keypoints.shape) == 2:
            # Один кадр: (25, 3)
            return self._augment_single_frame(keypoints, apply_all)
        elif len(keypoints.shape) == 3:
            # Последовательность: (N, 25, 3)
            return np.array([
                self._augment_single_frame(frame, apply_all)
                for frame in keypoints
            ])
        else:
            raise ValueError(f"Неожиданная форма keypoints: {keypoints.shape}")

    def _augment_single_frame(
        self, keypoints: np.ndarray, apply_all: bool = True
    ) -> np.ndarray:
        """Аугментировать один кадр."""
        augmented = keypoints.copy()
        
        if apply_all:
            # 1. Добавить шум (как в MediaPipe реальных видео)
            augmented = self._add_noise(augmented)
            
            # 2. Добавить асимметрию
            augmented = self._add_asymmetry(augmented)
            
            # 3. Добавить естественную изменчивость
            if self.add_natural_variation:
                augmented = self._add_natural_variation(augmented)
            
            # 4. Небольшое вращение
            augmented = self._apply_rotation(augmented)
            
            # 5. Небольшое масштабирование
            augmented = self._apply_scale(augmented)
        
        return augmented

    def _add_noise(self, keypoints: np.ndarray) -> np.ndarray:
        """Добавить шум к ключевым точкам."""
        noise = np.random.normal(0, self.noise_level, keypoints.shape)
        return keypoints + noise

    def _add_asymmetry(self, keypoints: np.ndarray) -> np.ndarray:
        """Добавить небольшую асимметрию (в реальности нет идеальной симметрии)."""
        augmented = keypoints.copy()
        
        # Случайная асимметрия
        asymmetry = np.random.uniform(
            self.asymmetry_range[0], self.asymmetry_range[1]
        )
        
        # Применяем к левой стороне
        for idx in self.left_side_indices:
            if idx < len(augmented):
                augmented[idx, :2] += asymmetry  # только x, y
        
        # Применяем к правой стороне (противоположное направление)
        for idx in self.right_side_indices:
            if idx < len(augmented):
                augmented[idx, :2] -= asymmetry
        
        return augmented

    def _add_natural_variation(self, keypoints: np.ndarray) -> np.ndarray:
        """Добавить естественную изменчивость движений."""
        augmented = keypoints.copy()
        
        # Небольшие случайные смещения для разных суставов
        for i in range(len(augmented)):
            variation = np.random.normal(0, 0.01, 3)  # очень маленькие изменения
            augmented[i] += variation
        
        return augmented

    def _apply_rotation(self, keypoints: np.ndarray) -> np.ndarray:
        """Применить небольшое случайное вращение."""
        # Случайный угол в радианах
        angle_deg = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.deg2rad(angle_deg)
        
        # Матрица вращения (только для x, y)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Центрируем перед вращением
        center = np.mean(keypoints, axis=0)
        centered = keypoints - center
        
        # Применяем вращение
        rotated = (rotation_matrix @ centered.T).T
        
        # Возвращаем обратно
        return rotated + center

    def _apply_scale(self, keypoints: np.ndarray) -> np.ndarray:
        """Применить небольшое случайное масштабирование."""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Центрируем перед масштабированием
        center = np.mean(keypoints, axis=0)
        centered = keypoints - center
        
        # Масштабируем
        scaled = centered * scale
        
        # Возвращаем обратно
        return scaled + center


def augment_mini_rgbd_to_realistic(
    keypoints: np.ndarray,
    noise_level: float = 0.02,
    asymmetry_range: tuple = (-0.03, 0.03),
) -> np.ndarray:
    """
    Добавляем "реалистичности" к синтетическим данным MINI-RGBD.
    
    Args:
        keypoints: Ключевые точки формы (25, 3) или (N, 25, 3)
        noise_level: Уровень шума
        asymmetry_range: Диапазон асимметрии
    
    Returns:
        Аугментированные ключевые точки
    """
    aug = DataAugmentation(
        noise_level=noise_level,
        asymmetry_range=asymmetry_range,
    )
    return aug.augment_keypoints(keypoints)
