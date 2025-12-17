
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
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.normalize = normalize
        self.normalize_relative_to = normalize_relative_to
        self.normalize_by_body = normalize_by_body
        self.rotate_to_canonical = rotate_to_canonical
        


        if target_hip_distance is None:


            self.target_hip_distance = 0.072221
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
        if keypoints is None or idx >= len(keypoints):
            return None
        return (keypoints[idx, 0], keypoints[idx, 1], keypoints[idx, 2])

    def _average_points(
        self,
        p1: Optional[Tuple[float, float, float]],
        p2: Optional[Tuple[float, float, float]],
    ) -> Optional[Tuple[float, float, float]]:
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
        if keypoints is None:
            return None


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


        hip_center = self._average_points(left_hip, right_hip)
        shoulder_center = self._average_points(left_shoulder, right_shoulder)
        left_upper_arm_mid = self._average_points(left_shoulder, left_elbow)
        right_upper_arm_mid = self._average_points(right_shoulder, right_elbow)


        joints = [
            hip_center,
            left_hip,
            right_hip,
            hip_center,
            left_knee,
            right_knee,
            shoulder_center,
            left_ankle,
            right_ankle,
            shoulder_center,
            left_foot_index,
            right_foot_index,
            shoulder_center,
            left_shoulder,
            right_shoulder,
            nose,
            left_upper_arm_mid,
            right_upper_arm_mid,
            left_elbow,
            right_elbow,
            left_wrist,
            right_wrist,
            left_index,
            right_index,
            nose,
        ]


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
        if not self.normalize or relative_to == "global":
            return pose

        normalized = pose.copy()
        

        if self.rotate_to_canonical:
            normalized = self._rotate_to_canonical_orientation(normalized)
        

        if self.normalize_by_body:
            normalized = self._normalize_by_body_bounding_box(normalized)
        else:

            normalized = self._normalize_by_torso(normalized)
        
        return normalized
    
    def _normalize_by_torso(self, pose: np.ndarray) -> np.ndarray:
        torso_center = (pose[1] + pose[2]) / 2
        normalized = pose.copy() - torso_center
        

        current_hip_distance = np.linalg.norm(pose[1] - pose[2])
        if current_hip_distance > 1e-6:
            normalized = normalized / current_hip_distance
        
        return normalized
    
    def _normalize_by_body_bounding_box(self, pose: np.ndarray) -> np.ndarray:

        bbox_min = np.min(pose, axis=0)
        bbox_max = np.max(pose, axis=0)
        

        center = (bbox_min + bbox_max) / 2
        centered = pose.copy() - center
        

        body_size = np.linalg.norm(bbox_max - bbox_min)
        if body_size > 1e-6:
            normalized = centered / body_size
        else:
            normalized = centered
        
        return normalized
    
    def _rotate_to_canonical_orientation(self, pose: np.ndarray) -> np.ndarray:



        

        pelvis = (pose[1] + pose[2]) / 2
        neck = (pose[13] + pose[14]) / 2 if len(pose) > 14 else (pose[1] + pose[2]) / 2
        
        spine_vector = neck[:2] - pelvis[:2]
        

        spine_norm = np.linalg.norm(spine_vector)
        if spine_norm > 1e-6:

            spine_vector = spine_vector / spine_norm
            


            target_vector = np.array([0.0, 1.0])
            

            cos_angle = np.dot(spine_vector, target_vector)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            

            cross_product = spine_vector[0] * target_vector[1] - spine_vector[1] * target_vector[0]
            if cross_product < 0:
                angle = -angle
            

            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            

            rotated_xy = (rotation_matrix @ pose[:, :2].T).T
            rotated = np.concatenate([rotated_xy, pose[:, 2:3]], axis=1)
            
            return rotated
        else:

            return pose

    def filter_invalid_poses(
        self, poses: List[Optional[np.ndarray]], min_valid_joints: int = 15
    ) -> List[np.ndarray]:
        valid_poses = []
        
        for pose in poses:
            if pose is None:
                continue
            

            valid_mask = np.any(pose != 0, axis=1)
            num_valid = np.sum(valid_mask)
            
            if num_valid >= min_valid_joints:
                valid_poses.append(pose)
        
        return valid_poses

    def create_sequences(
        self, poses: List[np.ndarray]
    ) -> List[np.ndarray]:
        sequences = []
        
        for i in range(0, len(poses) - self.sequence_length + 1, self.sequence_stride):
            sequence = poses[i : i + self.sequence_length]
            

            if len(sequence) == self.sequence_length:
                sequence_array = np.array(sequence, dtype=np.float32)
                sequences.append(sequence_array)
        
        return sequences

    def filter_low_confidence_keypoints(
        self, keypoints_array: np.ndarray, min_visibility: float = 0.3
    ) -> np.ndarray:
        filtered = keypoints_array.copy()
        visibility = keypoints_array[:, 3]
        

        low_confidence_mask = visibility < min_visibility
        filtered[low_confidence_mask, :3] = np.nan
        
        return filtered
    
    def smooth_keypoints_temporally(
        self, keypoints_list: List[np.ndarray], window_size: int = 3
    ) -> List[np.ndarray]:
        if len(keypoints_list) < window_size:
            return keypoints_list
        
        smoothed = []
        for i in range(len(keypoints_list)):

            start = max(0, i - window_size // 2)
            end = min(len(keypoints_list), i + window_size // 2 + 1)
            

            window = np.array(keypoints_list[start:end])
            smoothed_frame = np.nanmean(window, axis=0)
            

            smoothed_frame = np.nan_to_num(smoothed_frame, nan=0.0)
            smoothed.append(smoothed_frame.astype(np.float32))
        
        return smoothed
    
    def process_keypoints(
        self, keypoints_list: List[Optional[np.ndarray]]
    ) -> List[np.ndarray]:

        mini_rgbd_poses = []
        for kp in keypoints_list:
            pose = self.convert_to_mini_rgbd(kp)
            if pose is not None:
                mini_rgbd_poses.append(pose)
        
        if len(mini_rgbd_poses) == 0:
            logger.warning("Нет валидных поз после конвертации")
            return []
        

        valid_poses = self.filter_invalid_poses(mini_rgbd_poses)
        
        if len(valid_poses) == 0:
            logger.warning("Нет валидных поз после фильтрации")
            return []
        

        if self.normalize:
            normalized_poses = [
                self.normalize_pose(pose, self.normalize_relative_to)
                for pose in valid_poses
            ]
        else:
            normalized_poses = valid_poses
        

        sequences = self.create_sequences(normalized_poses)
        
        logger.info(
            f"Обработано: {len(keypoints_list)} кадров → "
            f"{len(mini_rgbd_poses)} конвертировано → "
            f"{len(valid_poses)} валидных → "
            f"{len(sequences)} последовательностей"
        )
        
        return sequences

    def flatten_sequence(self, sequence: np.ndarray) -> np.ndarray:
        return sequence.reshape(sequence.shape[0], -1)

