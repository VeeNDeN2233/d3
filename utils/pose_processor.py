"""
Обработчик ключевых точек позы.

Преобразует MediaPipe landmarks в MINI-RGBD формат,
нормализует относительно торса и создает последовательности.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from video_processor import (
    MINI_RGBD_JOINT_NAMES,
    MP_LEFT_ANKLE,
    MP_LEFT_ELBOW,
    MP_LEFT_FOOT_INDEX,
    MP_LEFT_HIP,
    MP_LEFT_INDEX,
    MP_LEFT_KNEE,
    MP_LEFT_SHOULDER,
    MP_LEFT_WRIST,
    MP_NOSE,
    MP_RIGHT_ANKLE,
    MP_RIGHT_ELBOW,
    MP_RIGHT_FOOT_INDEX,
    MP_RIGHT_HIP,
    MP_RIGHT_INDEX,
    MP_RIGHT_KNEE,
    MP_RIGHT_SHOULDER,
    MP_RIGHT_WRIST,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseProcessor:
    """
    Обработка ключевых точек:
    1. MediaPipe (33 точки) → MINI-RGBD формат (25 суставов)
    2. Нормализация относительно торса
    3. Создание последовательностей (30 кадров)
    4. Удаление шума и пропущенных точек
    """

    def __init__(
        self,
        sequence_length: int = 30,
        sequence_stride: int = 1,
        normalize: bool = True,
        normalize_relative_to: str = "torso",
        target_hip_distance: Optional[float] = None,
        normalize_by_body: bool = False,
        rotate_to_canonical: bool = False,
    ):
        """
        Инициализация процессора позы.
        
        Args:
            sequence_length: Длина последовательности в кадрах
            sequence_stride: Шаг между последовательностями
            normalize: Нормализовать координаты
            normalize_relative_to: Относительно чего нормализовать ("torso" или "global")
        """
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.normalize = normalize
        self.normalize_relative_to = normalize_relative_to
        self.normalize_by_body = normalize_by_body
        self.rotate_to_canonical = rotate_to_canonical
        
        # Фиксированный hip_distance из тренировочных данных
        # Используется для стандартизации нормализации
        if target_hip_distance is None:
            # Значение из тренировочных данных (mean из hip_distance_analysis.json)
            # Это hip_distance в нормализованных координатах MediaPipe [0, 1]
            self.target_hip_distance = 0.072221  # Среднее значение из тренировки
        else:
            self.target_hip_distance = target_hip_distance
        
        logger.info(
            f"Инициализирован PoseProcessor: seq_len={sequence_length}, "
            f"stride={sequence_stride}, normalize={normalize}, "
            f"relative_to={normalize_relative_to}, "
            f"normalize_by_body={normalize_by_body}, "
            f"rotate_to_canonical={rotate_to_canonical}"
        )
        
        logger.info(
            f"Инициализирован PoseProcessor: "
            f"seq_len={sequence_length}, stride={sequence_stride}, "
            f"normalize={normalize}, relative_to={normalize_relative_to}"
        )

    def _get_landmark_point(
        self, keypoints: np.ndarray, idx: int
    ) -> Optional[Tuple[float, float, float]]:
        """Получить координаты landmark по индексу из MediaPipe keypoints."""
        if keypoints is None or idx >= len(keypoints):
            return None
        return (keypoints[idx, 0], keypoints[idx, 1], keypoints[idx, 2])

    def _average_points(
        self,
        p1: Optional[Tuple[float, float, float]],
        p2: Optional[Tuple[float, float, float]],
    ) -> Optional[Tuple[float, float, float]]:
        """Вычислить среднюю точку между двумя точками."""
        if p1 is None and p2 is None:
            return None
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)

    def convert_to_mini_rgbd(
        self, keypoints: Optional[np.ndarray], width: int = 1, height: int = 1
    ) -> Optional[np.ndarray]:
        """
        Конвертировать MediaPipe Pose landmarks в формат MINI-RGBD (25 суставов).
        
        Args:
            keypoints: Массив MediaPipe keypoints формы (33, 4) [x, y, z, visibility]
            width: Ширина изображения (для денормализации, если нужно)
            height: Высота изображения (для денормализации, если нужно)
        
        Returns:
            Массив формы (25, 3) с координатами [x, y, z] или None
        """
        if keypoints is None:
            return None

        # Получаем основные точки
        nose = self._get_landmark_point(keypoints, MP_NOSE)
        left_shoulder = self._get_landmark_point(keypoints, MP_LEFT_SHOULDER)
        right_shoulder = self._get_landmark_point(keypoints, MP_RIGHT_SHOULDER)
        left_elbow = self._get_landmark_point(keypoints, MP_LEFT_ELBOW)
        right_elbow = self._get_landmark_point(keypoints, MP_RIGHT_ELBOW)
        left_wrist = self._get_landmark_point(keypoints, MP_LEFT_WRIST)
        right_wrist = self._get_landmark_point(keypoints, MP_RIGHT_WRIST)
        left_index = self._get_landmark_point(keypoints, MP_LEFT_INDEX)
        right_index = self._get_landmark_point(keypoints, MP_RIGHT_INDEX)
        left_hip = self._get_landmark_point(keypoints, MP_LEFT_HIP)
        right_hip = self._get_landmark_point(keypoints, MP_RIGHT_HIP)
        left_knee = self._get_landmark_point(keypoints, MP_LEFT_KNEE)
        right_knee = self._get_landmark_point(keypoints, MP_RIGHT_KNEE)
        left_ankle = self._get_landmark_point(keypoints, MP_LEFT_ANKLE)
        right_ankle = self._get_landmark_point(keypoints, MP_RIGHT_ANKLE)
        left_foot_index = self._get_landmark_point(keypoints, MP_LEFT_FOOT_INDEX)
        right_foot_index = self._get_landmark_point(keypoints, MP_RIGHT_FOOT_INDEX)

        # Вычисляем промежуточные точки
        hip_center = self._average_points(left_hip, right_hip)
        shoulder_center = self._average_points(left_shoulder, right_shoulder)
        left_upper_arm_mid = self._average_points(left_shoulder, left_elbow)
        right_upper_arm_mid = self._average_points(right_shoulder, right_elbow)

        # Строим массив из 25 суставов MINI-RGBD
        joints = [
            hip_center,  # 0: global
            left_hip,  # 1: leftThigh
            right_hip,  # 2: rightThigh
            hip_center,  # 3: spine
            left_knee,  # 4: leftCalf
            right_knee,  # 5: rightCalf
            shoulder_center,  # 6: spine1
            left_ankle,  # 7: leftFoot
            right_ankle,  # 8: rightFoot
            shoulder_center,  # 9: spine2
            left_foot_index,  # 10: leftToes
            right_foot_index,  # 11: rightToes
            shoulder_center,  # 12: neck
            left_shoulder,  # 13: leftShoulder
            right_shoulder,  # 14: rightShoulder
            nose,  # 15: head
            left_upper_arm_mid,  # 16: leftUpperArm
            right_upper_arm_mid,  # 17: rightUpperArm
            left_elbow,  # 18: leftForeArm
            right_elbow,  # 19: rightForeArm
            left_wrist,  # 20: leftHand
            right_wrist,  # 21: rightHand
            left_index,  # 22: leftFingers
            right_index,  # 23: rightFingers
            nose,  # 24: noseVertex
        ]

        # Конвертируем в numpy массив, заменяя None на нули
        joint_array = []
        for joint in joints:
            if joint is None:
                joint_array.append([0.0, 0.0, 0.0])
            else:
                joint_array.append(list(joint))

        return np.array(joint_array, dtype=np.float32)

    def normalize_pose(
        self, pose: np.ndarray, relative_to: str = "torso"
    ) -> np.ndarray:
        """
        Нормализовать позу относительно торса или глобально.
        
        Args:
            pose: Массив формы (25, 3) с координатами суставов
            relative_to: "torso" - относительно центра торса, "global" - без изменений
        
        Returns:
            Нормализованная поза формы (25, 3)
        """
        if not self.normalize or relative_to == "global":
            return pose

        normalized = pose.copy()
        
        # РЕШЕНИЕ 2: Вращение к канонической ориентации (если включено)
        if self.rotate_to_canonical:
            normalized = self._rotate_to_canonical_orientation(normalized)
        
        # РЕШЕНИЕ 1: Нормализация по bounding box всего тела (если включено)
        if self.normalize_by_body:
            normalized = self._normalize_by_body_bounding_box(normalized)
        else:
            # Стандартная нормализация относительно торса
            normalized = self._normalize_by_torso(normalized)
        
        return normalized
    
    def _normalize_by_torso(self, pose: np.ndarray) -> np.ndarray:
        """Нормализация относительно торса (бедра)."""
        torso_center = (pose[1] + pose[2]) / 2  # средняя точка между бедрами
        normalized = pose.copy() - torso_center
        
        # Масштабирование относительно размера торса
        current_hip_distance = np.linalg.norm(pose[1] - pose[2])
        if current_hip_distance > 1e-6:
            normalized = normalized / current_hip_distance
        
        return normalized
    
    def _normalize_by_body_bounding_box(self, pose: np.ndarray) -> np.ndarray:
        """
        Нормализация по bounding box ВСЕХ точек, а не только по бедрам.
        
        Это более устойчиво к разным масштабам и положениям тела.
        """
        # 1. Найти bounding box всех точек
        bbox_min = np.min(pose, axis=0)  # [min_x, min_y, min_z]
        bbox_max = np.max(pose, axis=0)  # [max_x, max_y, max_z]
        
        # 2. Центрирование
        center = (bbox_min + bbox_max) / 2
        centered = pose.copy() - center
        
        # 3. Масштабирование по диагонали bounding box
        body_size = np.linalg.norm(bbox_max - bbox_min)
        if body_size > 1e-6:
            normalized = centered / body_size
        else:
            normalized = centered
        
        return normalized
    
    def _rotate_to_canonical_orientation(self, pose: np.ndarray) -> np.ndarray:
        """
        Повернуть тело к канонической ориентации (горизонтально).
        
        Это помогает компенсировать разные углы съемки.
        """
        # Индексы для MINI-RGBD формата:
        # pose[1] = leftThigh, pose[2] = rightThigh
        # pose[13] = leftShoulder, pose[14] = rightShoulder
        
        # 1. Найти вектор "позвоночника" (от таза к шее)
        pelvis = (pose[1] + pose[2]) / 2  # средняя точка бедер
        neck = (pose[13] + pose[14]) / 2 if len(pose) > 14 else (pose[1] + pose[2]) / 2  # средняя точка плеч
        
        spine_vector = neck[:2] - pelvis[:2]  # вектор позвоночника (только x, y)
        
        # 2. Вычислить угол наклона
        spine_norm = np.linalg.norm(spine_vector)
        if spine_norm > 1e-6:
            # Нормализуем вектор
            spine_vector = spine_vector / spine_norm
            
            # Вычисляем угол относительно горизонтали (0, 1) - вертикальный вектор
            # Хотим, чтобы позвоночник был вертикальным
            target_vector = np.array([0.0, 1.0])  # вертикальный вектор вверх
            
            # Угол между spine_vector и target_vector
            cos_angle = np.dot(spine_vector, target_vector)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Определяем направление вращения
            cross_product = spine_vector[0] * target_vector[1] - spine_vector[1] * target_vector[0]
            if cross_product < 0:
                angle = -angle
            
            # 3. Повернуть ВСЕ точки на -angle (чтобы позвоночник стал вертикальным)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Применить вращение к x,y координатам
            rotated_xy = (rotation_matrix @ pose[:, :2].T).T
            rotated = np.concatenate([rotated_xy, pose[:, 2:3]], axis=1)
            
            return rotated
        else:
            # Если не можем определить позвоночник, возвращаем как есть
            return pose

    def filter_invalid_poses(
        self, poses: List[Optional[np.ndarray]], min_valid_joints: int = 15
    ) -> List[np.ndarray]:
        """
        Фильтровать невалидные позы.
        
        Args:
            poses: Список поз (каждая форма (25, 3) или None)
            min_valid_joints: Минимальное количество валидных суставов
        
        Returns:
            Список валидных поз
        """
        valid_poses = []
        
        for pose in poses:
            if pose is None:
                continue
            
            # Проверяем количество ненулевых суставов
            valid_mask = np.any(pose != 0, axis=1)
            num_valid = np.sum(valid_mask)
            
            if num_valid >= min_valid_joints:
                valid_poses.append(pose)
        
        return valid_poses

    def create_sequences(
        self, poses: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Создать последовательности заданной длины из поз.
        
        Args:
            poses: Список поз, каждая форма (25, 3)
        
        Returns:
            Список последовательностей, каждая форма (sequence_length, 25, 3)
        """
        sequences = []
        
        for i in range(0, len(poses) - self.sequence_length + 1, self.sequence_stride):
            sequence = poses[i : i + self.sequence_length]
            
            # Проверяем, что все позы в последовательности валидны
            if len(sequence) == self.sequence_length:
                sequence_array = np.array(sequence, dtype=np.float32)
                sequences.append(sequence_array)
        
        return sequences

    def filter_low_confidence_keypoints(
        self, keypoints_array: np.ndarray, min_visibility: float = 0.3
    ) -> np.ndarray:
        """
        Фильтрует точки с низкой уверенностью (visibility).
        Точки с visibility < min_visibility заменяются на NaN.
        
        Args:
            keypoints_array: Массив формы (33, 4) [x, y, z, visibility]
            min_visibility: Минимальный порог visibility
        
        Returns:
            Отфильтрованный массив (точки с низкой уверенностью = NaN)
        """
        filtered = keypoints_array.copy()
        visibility = keypoints_array[:, 3]  # 4-й столбец - visibility
        
        # Заменяем точки с низкой уверенностью на NaN
        low_confidence_mask = visibility < min_visibility
        filtered[low_confidence_mask, :3] = np.nan  # x, y, z становятся NaN
        
        return filtered
    
    def smooth_keypoints_temporally(
        self, keypoints_list: List[np.ndarray], window_size: int = 3
    ) -> List[np.ndarray]:
        """
        Сглаживает ключевые точки по времени с помощью скользящего среднего.
        
        Args:
            keypoints_list: Список массивов формы (33, 4)
            window_size: Размер окна для сглаживания
        
        Returns:
            Список сглаженных массивов
        """
        if len(keypoints_list) < window_size:
            return keypoints_list
        
        smoothed = []
        for i in range(len(keypoints_list)):
            # Окно для скользящего среднего
            start = max(0, i - window_size // 2)
            end = min(len(keypoints_list), i + window_size // 2 + 1)
            
            # Среднее по окну (игнорируем NaN)
            window = np.array(keypoints_list[start:end])
            smoothed_frame = np.nanmean(window, axis=0)
            
            # Заменяем NaN на 0
            smoothed_frame = np.nan_to_num(smoothed_frame, nan=0.0)
            smoothed.append(smoothed_frame.astype(np.float32))
        
        return smoothed
    
    def process_keypoints(
        self, keypoints_list: List[Optional[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Полный пайплайн обработки ключевых точек.
        
        Args:
            keypoints_list: Список MediaPipe keypoints (каждая (33, 4) или None)
        
        Returns:
            Список последовательностей формы (sequence_length, 25, 3)
        """
        # 1. Конвертируем в MINI-RGBD формат
        mini_rgbd_poses = []
        for kp in keypoints_list:
            pose = self.convert_to_mini_rgbd(kp)
            if pose is not None:
                mini_rgbd_poses.append(pose)
        
        if len(mini_rgbd_poses) == 0:
            logger.warning("Нет валидных поз после конвертации")
            return []
        
        # 2. Фильтруем невалидные позы
        valid_poses = self.filter_invalid_poses(mini_rgbd_poses)
        
        if len(valid_poses) == 0:
            logger.warning("Нет валидных поз после фильтрации")
            return []
        
        # 3. Нормализуем
        if self.normalize:
            normalized_poses = [
                self.normalize_pose(pose, self.normalize_relative_to)
                for pose in valid_poses
            ]
        else:
            normalized_poses = valid_poses
        
        # 4. Создаем последовательности
        sequences = self.create_sequences(normalized_poses)
        
        logger.info(
            f"Обработано: {len(keypoints_list)} кадров → "
            f"{len(mini_rgbd_poses)} конвертировано → "
            f"{len(valid_poses)} валидных → "
            f"{len(sequences)} последовательностей"
        )
        
        return sequences

    def flatten_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Преобразовать последовательность в плоский вектор для модели.
        
        Args:
            sequence: Массив формы (sequence_length, 25, 3)
        
        Returns:
            Массив формы (sequence_length, 75) = (sequence_length, 25*3)
        """
        return sequence.reshape(sequence.shape[0], -1)

