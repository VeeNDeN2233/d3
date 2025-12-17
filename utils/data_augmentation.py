
import logging
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:

    def __init__(
        self,
        noise_level: float = 0.02,
        asymmetry_range: tuple = (-0.03, 0.03),
        add_natural_variation: bool = True,
        rotation_range: float = 5.0,
        scale_range: tuple = (0.95, 1.05),
    ):
        self.noise_level = noise_level
        self.asymmetry_range = asymmetry_range
        self.add_natural_variation = add_natural_variation
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        


        self.left_side_indices = [
            1,
            4,
            7,
            10,
            13,
            16,
            18,
            19,
            21,
        ]
        
        self.right_side_indices = [
            2,
            5,
            8,
            11,
            14,
            17,
            19,
            20,
            22,
        ]
        

        self.right_side_indices = [
            2,
            5,
            8,
            11,
            14,
            17,
            20,
            21,
            23,
        ]
        
        logger.info(
            f"Инициализирован DataAugmentation: "
            f"noise={noise_level}, asymmetry={asymmetry_range}, "
            f"rotation={rotation_range}deg, scale={scale_range}"
        )

    def augment_keypoints(
        self, keypoints: np.ndarray, apply_all: bool = True
    ) -> np.ndarray:
        if len(keypoints.shape) == 2:

            return self._augment_single_frame(keypoints, apply_all)
        elif len(keypoints.shape) == 3:

            return np.array([
                self._augment_single_frame(frame, apply_all)
                for frame in keypoints
            ])
        else:
            raise ValueError(f"Неожиданная форма keypoints: {keypoints.shape}")

    def _augment_single_frame(
        self, keypoints: np.ndarray, apply_all: bool = True
    ) -> np.ndarray:
        augmented = keypoints.copy()
        
        if apply_all:

            augmented = self._add_noise(augmented)
            

            augmented = self._add_asymmetry(augmented)
            

            if self.add_natural_variation:
                augmented = self._add_natural_variation(augmented)
            

            augmented = self._apply_rotation(augmented)
            

            augmented = self._apply_scale(augmented)
        
        return augmented

    def _add_noise(self, keypoints: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_level, keypoints.shape)
        return keypoints + noise

    def _add_asymmetry(self, keypoints: np.ndarray) -> np.ndarray:
        augmented = keypoints.copy()
        

        asymmetry = np.random.uniform(
            self.asymmetry_range[0], self.asymmetry_range[1]
        )
        

        for idx in self.left_side_indices:
            if idx < len(augmented):
                augmented[idx, :2] += asymmetry
        

        for idx in self.right_side_indices:
            if idx < len(augmented):
                augmented[idx, :2] -= asymmetry
        
        return augmented

    def _add_natural_variation(self, keypoints: np.ndarray) -> np.ndarray:
        augmented = keypoints.copy()
        

        for i in range(len(augmented)):
            variation = np.random.normal(0, 0.01, 3)
            augmented[i] += variation
        
        return augmented

    def _apply_rotation(self, keypoints: np.ndarray) -> np.ndarray:

        angle_deg = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.deg2rad(angle_deg)
        

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        

        center = np.mean(keypoints, axis=0)
        centered = keypoints - center
        

        rotated = (rotation_matrix @ centered.T).T
        

        return rotated + center

    def _apply_scale(self, keypoints: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        

        center = np.mean(keypoints, axis=0)
        centered = keypoints - center
        

        scaled = centered * scale
        

        return scaled + center


def augment_mini_rgbd_to_realistic(
    keypoints: np.ndarray,
    noise_level: float = 0.02,
    asymmetry_range: tuple = (-0.03, 0.03),
) -> np.ndarray:
    aug = DataAugmentation(
        noise_level=noise_level,
        asymmetry_range=asymmetry_range,
    )
    return aug.augment_keypoints(keypoints)
