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
    # Используем более совместимый кодек для веб-браузеров
    fourcc_options = [
        ('avc1', 'H.264/AVC'),  # Самый совместимый для веб
        ('mp4v', 'MPEG-4'),      # Запасной вариант
        ('XVID', 'XVID'),        # Еще один вариант
    ]
    
    out = None
    used_codec = None
    for codec_name, codec_desc in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            if out.isOpened():
                used_codec = codec_desc
                logger.info(f"Используется кодек: {codec_desc} ({codec_name})")
                break
            else:
                out.release()
                out = None
        except Exception as e:
            logger.warning(f"Не удалось использовать кодек {codec_name}: {e}")
            if out:
                out.release()
                out = None
    
    if out is None or not out.isOpened():
        raise RuntimeError(f"Не удалось создать VideoWriter с доступными кодеками. Попробованы: {[c[0] for c in fourcc_options]}")
    
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
                    # Рисуем скелет напрямую на кадре, используя координаты
                    # Преобразуем нормализованные координаты в пиксели
                    for lm_data in landmarks_list:
                        x_norm = float(lm_data["x"])
                        y_norm = float(lm_data["y"])
                        visibility = float(lm_data.get("visibility", 1.0))
                        
                        # Преобразуем в пиксели
                        x_pixel = int(x_norm * width)
                        y_pixel = int(y_norm * height)
                        
                        # Рисуем точку только если visibility достаточно высокая
                        if visibility > 0.5:
                            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
                    
                    # Рисуем соединения между ключевыми точками
                    if len(landmarks_list) >= 33:
                        connections = [
                            (0, 1), (1, 2), (2, 3), (3, 7),
                            (11, 12), (11, 13), (13, 15),
                            (12, 14), (14, 16), (11, 23), (12, 24),
                            (23, 24), (23, 25), (25, 27),
                            (24, 26), (26, 28),
                        ]
                        
                        for start_idx, end_idx in connections:
                            if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                                lm1 = landmarks_list[start_idx]
                                lm2 = landmarks_list[end_idx]
                                
                                vis1 = float(lm1.get("visibility", 1.0))
                                vis2 = float(lm2.get("visibility", 1.0))
                                
                                if vis1 > 0.5 and vis2 > 0.5:
                                    x1 = int(float(lm1["x"]) * width)
                                    y1 = int(float(lm1["y"]) * height)
                                    x2 = int(float(lm2["x"]) * width)
                                    y2 = int(float(lm2["y"]) * height)
                                    
                                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
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
        if out:
            out.release()

    # Проверяем, что файл создан и имеет размер
    if not output_video_path.exists():
        raise RuntimeError(f"Видео файл не был создан: {output_video_path}")
    
    file_size = output_video_path.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Видео файл пуст: {output_video_path}")
    
    logger.info(f"Видео с скелетом создано: {output_video_path} ({frame_idx} кадров, {file_size / 1024 / 1024:.2f} MB, кодек: {used_codec})")
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
    logger.info(f"Всего keypoints для {len(keypoints_list)} кадров")
    
    # Проверяем, сколько кадров с keypoints
    valid_keypoints_count = sum(1 for kp in keypoints_list if kp is not None)
    logger.info(f"Валидных keypoints: {valid_keypoints_count} из {len(keypoints_list)}")
    
    # Открываем исходное видео
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Видео: {width}x{height} @ {fps} FPS")
    
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
            if frame_idx < len(keypoints_list):
                keypoints = keypoints_list[frame_idx]  # (33, 4) [x, y, z, visibility]
                
                # Проверяем, что keypoints не пустой и имеет правильную форму
                # Теперь keypoints всегда np.array, никогда не None
                if keypoints is not None and len(keypoints) > 0 and keypoints.shape[0] == 33:
                    # Проверяем форму keypoints
                    if frame_idx == 0:
                        logger.info(f"Форма keypoints для первого кадра: {keypoints.shape}")
                        logger.info(f"Пример координат первого keypoint: x={keypoints[0][0]:.4f}, y={keypoints[0][1]:.4f}")
                    
                    # Рисуем скелет напрямую на кадре, используя координаты
                    # Преобразуем нормализованные координаты в пиксели
                    for kp in keypoints:
                        # MediaPipe использует нормализованные координаты [0, 1]
                        x_norm, y_norm = float(kp[0]), float(kp[1])
                        
                        # Проверяем, что координаты в правильном диапазоне
                        # Если координаты в пикселях (больше 1.0), нормализуем
                        if x_norm > 1.0 or y_norm > 1.0:
                            x_norm = x_norm / width
                            y_norm = y_norm / height
                        
                        # Убеждаемся, что координаты в диапазоне [0, 1]
                        x_norm = max(0.0, min(1.0, x_norm))
                        y_norm = max(0.0, min(1.0, y_norm))
                        
                        # Преобразуем в пиксели для рисования
                        x_pixel = int(x_norm * width)
                        y_pixel = int(y_norm * height)
                        visibility = float(kp[3]) if len(kp) > 3 else 1.0
                        
                        # Рисуем точку только если visibility достаточно высокая
                        if visibility > 0.5:
                            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
                    
                    # Рисуем соединения между ключевыми точками вручную
                    # Используем POSE_CONNECTIONS для определения соединений
                    if len(keypoints) >= 33:  # MediaPipe Pose имеет 33 точки
                        # Основные соединения MediaPipe Pose
                        connections = [
                            (0, 1), (1, 2), (2, 3), (3, 7),  # Лицо
                            (11, 12),  # Плечи
                            (11, 13), (13, 15),  # Левая рука
                            (12, 14), (14, 16),  # Правая рука
                            (11, 23), (12, 24),  # Плечи к бедрам
                            (23, 24),  # Бедра
                            (23, 25), (25, 27),  # Левая нога
                            (24, 26), (26, 28),  # Правая нога
                        ]
                        
                        for start_idx, end_idx in connections:
                            if start_idx < len(keypoints) and end_idx < len(keypoints):
                                kp1 = keypoints[start_idx]
                                kp2 = keypoints[end_idx]
                                
                                # Проверяем visibility
                                vis1 = float(kp1[3]) if len(kp1) > 3 else 1.0
                                vis2 = float(kp2[3]) if len(kp2) > 3 else 1.0
                                
                                if vis1 > 0.5 and vis2 > 0.5:
                                    x1_norm, y1_norm = float(kp1[0]), float(kp1[1])
                                    x2_norm, y2_norm = float(kp2[0]), float(kp2[1])
                                    
                                    # Нормализуем если нужно
                                    if x1_norm > 1.0 or y1_norm > 1.0:
                                        x1_norm, y1_norm = x1_norm / width, y1_norm / height
                                    if x2_norm > 1.0 or y2_norm > 1.0:
                                        x2_norm, y2_norm = x2_norm / width, y2_norm / height
                                    
                                    # Убеждаемся в диапазоне [0, 1]
                                    x1_norm = max(0.0, min(1.0, x1_norm))
                                    y1_norm = max(0.0, min(1.0, y1_norm))
                                    x2_norm = max(0.0, min(1.0, x2_norm))
                                    y2_norm = max(0.0, min(1.0, y2_norm))
                                    
                                    x1, y1 = int(x1_norm * width), int(y1_norm * height)
                                    x2, y2 = int(x2_norm * width), int(y2_norm * height)
                                    
                                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if frame_idx == 0:
                        logger.info(f"✅ Скелет нарисован для первого кадра ({len(keypoints)} точек)")
                else:
                    if frame_idx < 5:  # Логируем только первые несколько кадров
                        logger.warning(f"Пустые keypoints для кадра {frame_idx}")
            else:
                if frame_idx < 5:  # Логируем только первые несколько кадров
                    logger.debug(f"Нет keypoints для кадра {frame_idx}")
            
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
        if out:
            out.release()

    # Проверяем, что файл создан и имеет размер
    if not output_video_path.exists():
        raise RuntimeError(f"Видео файл не был создан: {output_video_path}")
    
    file_size = output_video_path.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Видео файл пуст: {output_video_path}")
    
    logger.info(f"Видео с скелетом создано: {output_video_path} ({frame_idx} кадров, {file_size / 1024 / 1024:.2f} MB, кодек: {used_codec})")
    return output_video_path

