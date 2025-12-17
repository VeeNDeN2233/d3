
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MiniRGBDDataLoader:

    def __init__(
        self,
        data_root: str,
        train_sequences: List[int],
        val_sequences: List[int],
        test_sequences: List[int],
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.data_root = Path(data_root)
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        logger.info(f"Инициализирован загрузчик данных из {data_root}")
        logger.info(f"Train sequences: {train_sequences}")
        logger.info(f"Val sequences: {val_sequences}")
        logger.info(f"Test sequences: {test_sequences}")

    def _get_sequence_path(self, sequence_num: int) -> Path:
        seq_str = f"{sequence_num:02d}"
        return self.data_root / seq_str / "rgb"

    def _load_rgb_image(self, image_path: Path) -> Optional[np.ndarray]:
        if not image_path.exists():
            logger.warning(f"Изображение не найдено: {image_path}")
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Не удалось загрузить изображение: {image_path}")
            return None
        

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _extract_keypoints(self, image: np.ndarray) -> Optional[np.ndarray]:
        results = self.pose.process(image)
        
        if results.pose_landmarks is None:
            return None
        
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([
                landmark.x,
                landmark.y,
                landmark.z,
                getattr(landmark, 'visibility', 0.0),
            ])
        
        return np.array(keypoints, dtype=np.float32)

    def load_sequence(
        self, sequence_num: int, max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        rgb_dir = self._get_sequence_path(sequence_num)
        
        if not rgb_dir.exists():
            logger.error(f"Директория не найдена: {rgb_dir}")
            return [], []
        

        image_files = sorted(rgb_dir.glob("syn_*.png"))
        
        if max_frames is not None:
            image_files = image_files[:max_frames]
        
        logger.info(f"Загрузка последовательности {sequence_num:02d}: {len(image_files)} кадров")
        
        images = []
        keypoints_list = []
        
        for img_path in image_files:

            img = self._load_rgb_image(img_path)
            if img is None:
                continue
            
            images.append(img)
            

            kp = self._extract_keypoints(img)
            keypoints_list.append(kp)
        
        logger.info(
            f"Загружено {len(images)} изображений, "
            f"найдено поз: {sum(1 for kp in keypoints_list if kp is not None)}"
        )
        
        return images, keypoints_list

    def load_train_data(
        self, max_frames_per_seq: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        logger.info("Загрузка данных для обучения...")
        
        all_images = []
        all_keypoints = []
        
        for seq_num in self.train_sequences:
            images, keypoints = self.load_sequence(seq_num, max_frames_per_seq)
            all_images.extend(images)
            all_keypoints.extend(keypoints)
        
        logger.info(f"Всего загружено для обучения: {len(all_images)} кадров")
        return all_images, all_keypoints

    def load_val_data(
        self, max_frames_per_seq: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        logger.info("Загрузка данных для валидации...")
        
        all_images = []
        all_keypoints = []
        
        for seq_num in self.val_sequences:
            images, keypoints = self.load_sequence(seq_num, max_frames_per_seq)
            all_images.extend(images)
            all_keypoints.extend(keypoints)
        
        logger.info(f"Всего загружено для валидации: {len(all_keypoints)} кадров")
        return all_images, all_keypoints

    def load_test_data(
        self, max_frames_per_seq: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        logger.info("Загрузка данных для теста...")
        
        all_images = []
        all_keypoints = []
        
        for seq_num in self.test_sequences:
            images, keypoints = self.load_sequence(seq_num, max_frames_per_seq)
            all_images.extend(images)
            all_keypoints.extend(keypoints)
        
        logger.info(f"Всего загружено для теста: {len(all_keypoints)} кадров")
        return all_images, all_keypoints

    def get_sequence_info(self) -> Dict[str, List[int]]:
        return {
            "train": self.train_sequences,
            "val": self.val_sequences,
            "test": self.test_sequences,
        }

    def __del__(self):
        try:
            if hasattr(self, 'pose'):
                self.pose.close()
        except Exception:
            pass

