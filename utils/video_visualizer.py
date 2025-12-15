"""
Визуализация видео с наложенным скелетом MediaPipe.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_skeleton_video(
    input_video_path: Path,
    keypoints_json_path: Path,
    output_video_path: Path,
    highlight_anomalies: Optional[List[bool]] = None,
    anomaly_errors: Optional[List[float]] = None,
    threshold: Optional[float] = None,
) -> Path:
    """
    Создать видео с наложенным скелетом MediaPipe.
    
    Args:
        input_video_path: Путь к исходному видео
        keypoints_json_path: Путь к JSON с ключевыми точками
        output_video_path: Путь для сохранения результата
        highlight_anomalies: Список флагов аномалий для каждого кадра (опционально)
        anomaly_errors: Список ошибок реконструкции (опционально)
        threshold: Порог аномалии (опционально)
    
    Returns:
        Путь к созданному видео
    """
    logger.info(f"Создание видео с скелетом: {output_video_path}")
    
    # Загружаем ключевые точки
    with open(keypoints_json_path, "r", encoding="utf-8") as f:
        keypoints_data = json.load(f)
    
    # Открываем исходное видео
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаем выходное видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Инициализация MediaPipe для визуализации
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    frame_idx = 0
    frames_data = keypoints_data.get("frames", [])
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Конвертируем BGR в RGB для MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Получаем ключевые точки для этого кадра
            if frame_idx < len(frames_data):
                frame_data = frames_data[frame_idx]
                landmarks_list = frame_data.get("landmarks")
                
                if landmarks_list:
                    # Создаем объект landmarks для MediaPipe
                    landmarks = mp_pose.PoseLandmark
                    pose_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                    
                    for lm_data in landmarks_list:
                        landmark = pose_landmarks.landmark.add()
                        landmark.x = lm_data["x"]
                        landmark.y = lm_data["y"]
                        landmark.z = lm_data.get("z", 0.0)
                        landmark.visibility = lm_data.get("visibility", 1.0)
                    
                    # Рисуем скелет на кадре
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    
                    # Если есть информация об аномалиях, добавляем визуализацию
                    if highlight_anomalies is not None and frame_idx < len(highlight_anomalies):
                        if highlight_anomalies[frame_idx]:
                            # Добавляем красную рамку для аномальных кадров
                            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)
                            
                            # Добавляем текст с ошибкой
                            if anomaly_errors is not None and frame_idx < len(anomaly_errors):
                                error_text = f"Anomaly: {anomaly_errors[frame_idx]:.4f}"
                                cv2.putText(
                                    frame, error_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                                )
                    
                    # Добавляем информацию о кадре
                    frame_text = f"Frame: {frame_idx}"
                    cv2.putText(
                        frame, frame_text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
            
            out.write(frame)
            frame_idx += 1
    
    finally:
        cap.release()
        out.release()
    
    logger.info(f"Видео с скелетом создано: {output_video_path} ({frame_idx} кадров)")
    return output_video_path


def create_skeleton_video_from_processed(
    input_video_path: Path,
    keypoints_list: List[Optional[np.ndarray]],
    output_video_path: Path,
    errors: Optional[List[float]] = None,
    is_anomaly: Optional[List[bool]] = None,
    threshold: Optional[float] = None,
) -> Path:
    """
    Создать видео с скелетом из уже обработанных ключевых точек.
    
    Args:
        input_video_path: Путь к исходному видео
        keypoints_list: Список массивов ключевых точек (33, 4) для каждого кадра
        output_video_path: Путь для сохранения результата
        errors: Список ошибок реконструкции (опционально)
        is_anomaly: Список флагов аномалий (опционально)
        threshold: Порог аномалии (опционально)
    
    Returns:
        Путь к созданному видео
    """
    logger.info(f"Создание видео с скелетом из обработанных точек: {output_video_path}")
    
    # Открываем исходное видео
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаем выходное видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Инициализация MediaPipe для визуализации
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
                # Получаем ключевые точки для этого кадра
                if frame_idx < len(keypoints_list) and keypoints_list[frame_idx] is not None:
                    keypoints = keypoints_list[frame_idx]  # (33, 4) [x, y, z, visibility]
                    
                    # Создаем объект landmarks для MediaPipe
                    pose_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                    
                    for kp in keypoints:
                        landmark = pose_landmarks.landmark.add()
                        # MediaPipe использует нормализованные координаты [0, 1]
                        # keypoints_list содержит нормализованные координаты из MediaPipe
                        # Координаты уже в формате [0, 1], но нужно убедиться
                        x, y = float(kp[0]), float(kp[1])
                        # Если координаты в пикселях, нормализуем
                        if x > 1.0 or y > 1.0:
                            x = x / width
                            y = y / height
                        landmark.x = x
                        landmark.y = y
                        landmark.z = float(kp[2]) if len(kp) > 2 else 0.0
                        landmark.visibility = float(kp[3]) if len(kp) > 3 else 1.0
                    
                    # Рисуем скелет на кадре
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                
                # Если есть информация об аномалиях, добавляем визуализацию
                if is_anomaly is not None and frame_idx < len(is_anomaly):
                    # Определяем, является ли этот кадр частью аномальной последовательности
                    # (используем ближайшую последовательность)
                    seq_idx = frame_idx // 30  # Примерно, если последовательности по 30 кадров
                    if seq_idx < len(is_anomaly) and is_anomaly[seq_idx]:
                        # Добавляем красную рамку для аномальных кадров
                        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 3)
                        
                        # Добавляем текст с ошибкой
                        if errors is not None and seq_idx < len(errors):
                            error_text = f"Anomaly: {errors[seq_idx]:.4f}"
                            cv2.putText(
                                frame, error_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                            )
                
                # Добавляем информацию о кадре
                frame_text = f"Frame: {frame_idx}"
                cv2.putText(
                    frame, frame_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
            
            out.write(frame)
            frame_idx += 1
    
    finally:
        cap.release()
        out.release()
    
    logger.info(f"Видео с скелетом создано: {output_video_path} ({frame_idx} кадров)")
    return output_video_path

