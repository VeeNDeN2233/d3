"""
Вспомогательные функции для анализа видео в Flask приложении.
"""

# Устанавливаем backend для matplotlib перед импортами
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

from inference_advanced import (
    load_model_and_detector,
    process_video,
    visualize_results,
    generate_report as generate_medical_report,
)
from utils.pose_processor import PoseProcessor
from utils.video_visualizer import create_skeleton_video_from_processed
from video_processor import VideoProcessor
from models.anomaly_detector import AnomalyDetector
from models.autoencoder_advanced import BidirectionalLSTMAutoencoder
import yaml
import torch

logger = logging.getLogger(__name__)


# Глобальные переменные для моделей
_model: Optional[BidirectionalLSTMAutoencoder] = None
_detector: Optional[AnomalyDetector] = None
_config: Optional[dict] = None
_video_processor: Optional[VideoProcessor] = None
_pose_processor: Optional[PoseProcessor] = None


def load_models_for_flask(config_path: str = "config.yaml", 
                          checkpoint_path: str = "checkpoints/best_model_advanced.pt") -> Tuple[bool, str]:
    """
    Загрузить модели для Flask приложения.
    
    Returns:
        Tuple (успех, сообщение)
    """
    global _model, _detector, _config, _video_processor, _pose_processor
    
    try:
        if _model is not None and _detector is not None:
            return True, "Модели уже загружены"
        
        # Загружаем конфигурацию
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        
        # Проверяем GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            return False, "GPU недоступен"
        
        # Загружаем модель и детектор
        _model, _detector = load_model_and_detector(
            Path(checkpoint_path), _config, device, model_type="bidir_lstm"
        )
        
        # Инициализация процессоров
        _video_processor = VideoProcessor(
            model_complexity=_config["pose"]["model_complexity"],
            min_detection_confidence=_config["pose"]["min_detection_confidence"],
            min_tracking_confidence=_config["pose"]["min_tracking_confidence"],
        )
        
        _pose_processor = PoseProcessor(
            sequence_length=_config["pose"]["sequence_length"],
            sequence_stride=_config["pose"]["sequence_stride"],
            normalize=_config["pose"]["normalize"],
            normalize_relative_to=_config["pose"]["normalize_relative_to"],
            target_hip_distance=_config["pose"].get("target_hip_distance"),
            normalize_by_body=_config["pose"].get("normalize_by_body", False),
            rotate_to_canonical=_config["pose"].get("rotate_to_canonical", False),
        )
        
        status_msg = f"Модели загружены успешно. (Bidirectional LSTM + Attention)\nGPU: {torch.cuda.get_device_name(0)}\nПорог: {_detector.threshold:.6f}"
        
        return True, status_msg
        
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}", exc_info=True)
        return False, f"Ошибка загрузки: {str(e)}"


def analyze_video_flask(
    video_path: Path,
    age_weeks: int = 12,
    gestational_age_weeks: int = 40,
    output_dir: Optional[Path] = None
) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Анализ видео для Flask приложения.
    
    Args:
        video_path: Путь к видеофайлу
        age_weeks: Возраст ребенка в неделях
        gestational_age_weeks: Гестационный возраст
        output_dir: Директория для сохранения результатов
    
    Returns:
        Tuple (успех, plot_path, video_path, report_text, error_message, output_dir_path)
    """
    global _model, _detector, _config, _video_processor, _pose_processor
    
    if _model is None or _detector is None:
        return False, None, None, None, "Модели не загружены"
    
    if not video_path.exists():
        return False, None, None, None, "Видео файл не найден"
    
    try:
        # Создаем директорию для результатов
        if output_dir is None:
            output_dir = Path("results") / f"analysis_{video_path.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Обрабатываем видео
        keypoints_list, errors, is_anomaly, sequences_array = process_video(
            video_path, _video_processor, _pose_processor, _detector, _config
        )
        
        # Визуализация
        visualize_results(errors, is_anomaly, output_dir, video_path.stem, _detector.threshold)
        
        # Создаем видео с наложенным скелетом
        skeleton_video_path = output_dir / "video_with_skeleton.mp4"
        try:
            if not keypoints_list or len(keypoints_list) == 0:
                logger.warning("keypoints_list пуст, невозможно создать видео с скелетом")
                skeleton_video_path = None
            else:
                logger.info(f"Создание видео с скелетом из {len(keypoints_list)} кадров")
                create_skeleton_video_from_processed(
                    video_path,
                    keypoints_list,
                    skeleton_video_path,
                    errors=errors,
                    is_anomaly=is_anomaly,
                    threshold=_detector.threshold
                )
                logger.info(f"Видео с скелетом создано: {skeleton_video_path}")
        except Exception as e:
            logger.error(f"Не удалось создать видео с скелетом: {e}", exc_info=True)
            skeleton_video_path = None
        
        # Генерация медицинского отчета
        report = generate_medical_report(
            video_path, errors, is_anomaly, _detector, output_dir,
            age_weeks=age_weeks, 
            gestational_age_weeks=gestational_age_weeks,
            sequences_array=sequences_array
        )
        
        # Форматируем отчет
        from utils.report_formatter import format_medical_report
        report_text = format_medical_report(report)
        
        # Пути к результатам
        plot_path = output_dir / "reconstruction_error.png"
        
        # Возвращаем относительные пути от results/ для Flask
        results_base = Path("results").resolve()
        output_dir_resolved = output_dir.resolve()
        
        try:
            # Получаем относительный путь от results/
            plot_relative = plot_path.resolve().relative_to(results_base)
            video_relative = skeleton_video_path.resolve().relative_to(results_base) if skeleton_video_path else None
        except ValueError:
            # Если не удается получить относительный путь, используем имя файла
            plot_relative = plot_path.name
            video_relative = skeleton_video_path.name if skeleton_video_path else None
        
        # Преобразуем в строку с правильными разделителями для URL
        plot_url_path = str(plot_relative).replace('\\', '/')
        video_url_path = str(video_relative).replace('\\', '/') if video_relative else None
        
        # Возвращаем относительный путь к директории результатов для скачивания
        try:
            output_dir_relative = output_dir.resolve().relative_to(results_base)
            output_dir_path = str(output_dir_relative).replace('\\', '/')
        except ValueError:
            output_dir_path = output_dir.name
        
        return True, plot_url_path, video_url_path, report_text, None, output_dir_path
        
    except Exception as e:
        logger.error(f"Ошибка анализа видео: {e}", exc_info=True)
        return False, None, None, None, f"Ошибка анализа: {str(e)}", None


def get_models_status() -> Dict[str, Any]:
    """Получить статус моделей."""
    return {
        'loaded': _model is not None and _detector is not None,
        'model': _model is not None,
        'detector': _detector is not None,
    }
