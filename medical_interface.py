"""
Медицинский интерфейс для анализа видео младенцев.

Gradio интерфейс для загрузки видео и получения медицинского отчета.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import yaml

from inference_advanced import (
    generate_report as generate_medical_report,
    load_model_and_detector,
    process_video,
    visualize_results,
)
from models.anomaly_detector import AnomalyDetector
from models.autoencoder_advanced import BidirectionalLSTMAutoencoder
from utils.pose_processor import PoseProcessor
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные для загруженных моделей
_model: Optional[BidirectionalLSTMAutoencoder] = None
_detector: Optional[AnomalyDetector] = None
_config: Optional[dict] = None
_video_processor: Optional[VideoProcessor] = None
_pose_processor: Optional[PoseProcessor] = None


def load_models(config_path: str = "config.yaml", checkpoint_path: str = "checkpoints/best_model_advanced.pt"):
    """Загрузить модели один раз при старте."""
    global _model, _detector, _config, _video_processor, _pose_processor
    
    if _model is not None:
        return "Модели уже загружены"
    
    try:
        # Загружаем конфигурацию
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        
        # Проверяем GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            return "Ошибка: GPU недоступен!"
        
        # Загружаем модель и детектор (улучшенная модель по умолчанию)
        checkpoint = Path(checkpoint_path)
        _model, _detector = load_model_and_detector(checkpoint, _config, device, model_type="bidir_lstm")
        
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
        
        return f"✅ Модели загружены успешно! (Bidirectional LSTM + Attention)\nGPU: {torch.cuda.get_device_name(0)}\nПорог: {_detector.threshold:.6f}"
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}", exc_info=True)
        return f"❌ Ошибка загрузки: {str(e)}"


def analyze_baby_video(video_file) -> Tuple[Optional[str], Optional[str]]:
    """
    Основная функция анализа видео.
    
    Args:
        video_file: Файл от Gradio File компонента
    
    Returns:
        Tuple (anomaly_plot_path, report_json)
    """
    global _model, _detector, _config, _video_processor, _pose_processor
    
    if _model is None or _detector is None:
        return None, "❌ Ошибка: Модели не загружены!\n\nНажмите 'Загрузить модели' для инициализации системы."
    
    try:
        if video_file is None:
            return None, "❌ Ошибка: Видео не загружено!\n\nПожалуйста, загрузите видео файл перед анализом."
        
        # Обработка разных типов входных данных от Gradio
        logger.info(f"Получен video_file типа: {type(video_file)}")
        
        # Gradio File может вернуть:
        # 1. Объект File с атрибутом .name
        # 2. Строку с путем
        # 3. Список файлов
        # 4. None
        
        actual_path = None
        
        # Если это список
        if isinstance(video_file, list):
            if len(video_file) > 0:
                video_file = video_file[0]
            else:
                return None, "❌ Ошибка: Список файлов пуст!"
        
        # Получаем путь к файлу
        if hasattr(video_file, 'name'):
            # Объект File от Gradio
            actual_path = video_file.name
            logger.info(f"Файл из объекта File: {actual_path}")
        elif isinstance(video_file, str):
            # Строка с путем
            actual_path = video_file.strip()
            logger.info(f"Файл из строки: {actual_path}")
        elif video_file is not None:
            # Попытка преобразовать в строку
            actual_path = str(video_file).strip()
            logger.info(f"Файл преобразован в строку: {actual_path}")
        
        if not actual_path or actual_path == "None":
            return None, "❌ Ошибка: Не удалось определить путь к файлу!\n\nПопробуйте загрузить видео снова."
        
        # Нормализуем путь (исправляем обратные слэши на Windows)
        actual_path = Path(actual_path).resolve()
        logger.info(f"Обработка файла: {actual_path}")
        
        # Проверяем существование файла
        if not actual_path.exists():
            logger.error(f"Файл не существует: {actual_path}")
            return None, f"❌ Ошибка: Файл не найден!\n\nПуть: {actual_path}\n\nПопробуйте загрузить видео снова."
        
        if not actual_path.is_file():
            logger.error(f"Путь не является файлом: {actual_path}")
            return None, f"❌ Ошибка: Указанный путь не является файлом!\n\nПуть: {actual_path}"
        
        # Обработка видео
        keypoints_list, errors, is_anomaly = process_video(
            actual_path, _video_processor, _pose_processor, _detector, _config
        )
        
        # Создаем временную директорию для результатов
        output_dir = Path("results") / f"analysis_{actual_path.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Визуализация
        visualize_results(errors, is_anomaly, output_dir, actual_path.stem, _detector.threshold)
        
        # Генерация медицинского отчета
        report = generate_medical_report(
            actual_path, errors, is_anomaly, _detector, output_dir
        )
        
        # Пути к результатам
        plot_path = output_dir / "reconstruction_error.png"
        
        # Форматирование отчета для отображения
        report_text = format_medical_report(report)
        
        # Возвращаем путь к графику аномалий и отчет
        return (
            str(plot_path) if plot_path.exists() else None,
            report_text,
        )
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        error_msg = f"❌ Ошибка при анализе видео:\n\n{str(e)}\n\n"
        error_msg += f"Тип ошибки: {type(e).__name__}\n\n"
        error_msg += "Пожалуйста, проверьте:\n"
        error_msg += "1. Видео файл загружен корректно\n"
        error_msg += "2. Модели загружены (нажмите 'Загрузить модели')\n"
        error_msg += "3. Формат видео поддерживается\n"
        return None, error_msg


def format_medical_report(report: Dict) -> str:
    """Форматировать медицинский отчет для отображения."""
    if not report:
        return "Ошибка: Отчет пуст"
    
    lines = []
    lines.append("=" * 60)
    lines.append("МЕДИЦИНСКИЙ ОТЧЕТ")
    lines.append("=" * 60)
    lines.append("")
    
    # Статистика
    stats = report.get("statistics", {})
    lines.append("СТАТИСТИКА:")
    lines.append(f"  Последовательностей: {stats.get('total_sequences', 'N/A')}")
    lines.append(f"  Аномальных: {stats.get('anomalous_sequences', 'N/A')} ({stats.get('anomaly_rate', 0):.2f}%)")
    lines.append("")
    
    # Ошибки реконструкции
    errors = report.get("reconstruction_errors", {})
    lines.append("ОШИБКИ РЕКОНСТРУКЦИИ:")
    lines.append(f"  Средняя: {errors.get('mean', 'N/A'):.6f}" if isinstance(errors.get('mean'), (int, float)) else f"  Средняя: {errors.get('mean', 'N/A')}")
    lines.append(f"  Максимальная: {errors.get('max', 'N/A'):.6f}" if isinstance(errors.get('max'), (int, float)) else f"  Максимальная: {errors.get('max', 'N/A')}")
    lines.append(f"  Минимальная: {errors.get('min', 'N/A'):.6f}" if isinstance(errors.get('min'), (int, float)) else f"  Минимальная: {errors.get('min', 'N/A')}")
    lines.append(f"  Стандартное отклонение: {errors.get('std', 'N/A'):.6f}" if isinstance(errors.get('std'), (int, float)) else f"  Стандартное отклонение: {errors.get('std', 'N/A')}")
    lines.append("")
    
    # Детекция аномалий
    anomaly = report.get("anomaly_detection", {})
    risk_level = anomaly.get("risk_level", "unknown")
    risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "unknown": "⚪"}
    
    lines.append("ДЕТЕКЦИЯ АНОМАЛИЙ:")
    lines.append(f"  Уровень риска: {risk_emoji.get(risk_level, '⚪')} {risk_level.upper()}")
    lines.append(f"  Порог аномалии: {anomaly.get('threshold', 'N/A'):.6f}" if isinstance(anomaly.get('threshold'), (int, float)) else f"  Порог аномалии: {anomaly.get('threshold', 'N/A')}")
    lines.append(f"  Средний score: {anomaly.get('mean_anomaly_score', 'N/A'):.6f}" if isinstance(anomaly.get('mean_anomaly_score'), (int, float)) else f"  Средний score: {anomaly.get('mean_anomaly_score', 'N/A')}")
    lines.append(f"  Процент аномалий: {anomaly.get('anomaly_rate_percent', 0):.2f}%")
    lines.append("")
    
    # Рекомендации
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append("РЕКОМЕНДАЦИИ:")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def create_medical_interface():
    """Создать Gradio интерфейс."""
    
    with gr.Blocks(title="Детектор аномалий движений младенцев", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🍼 Детектор аномалий движений младенцев
            
            ### Используется улучшенная модель: **Bidirectional LSTM + Attention**
            
            Система для анализа движений младенцев и оценки риска нарушений моторики на основе RGB-видео.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Шаг 1: Инициализация")
                model_status = gr.Textbox(
                    label="Статус моделей",
                    value="⏳ Нажмите 'Загрузить модели' для инициализации системы",
                    interactive=False,
                    lines=3,
                )
                load_models_btn = gr.Button(
                    "🔄 Загрузить модели", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Шаг 2: Загрузка видео")
                video_input = gr.File(
                    label="Выберите видео файл",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                    file_count="single",
                    height=100,
                )
                gr.Markdown("**Поддерживаемые форматы:** MP4, AVI, MOV, MKV, WEBM")
        
        gr.Markdown("---")
        
        with gr.Row():
            analyze_btn = gr.Button(
                "🚀 Анализировать видео", 
                variant="primary",
                size="lg",
                scale=1
            )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 График ошибки реконструкции")
                anomaly_plot = gr.Image(
                    label="График аномалий",
                    height=400
                )
            
            with gr.Column():
                gr.Markdown("### 📄 Медицинский отчет")
                report_output = gr.Textbox(
                    label="Результаты анализа",
                    lines=25,
                    max_lines=30,
                    interactive=False,
                )
        
        # Обработчики событий
        load_models_btn.click(
            fn=load_models,
            outputs=model_status,
        )
        
        analyze_btn.click(
            fn=analyze_baby_video,
            inputs=video_input,  # Используем файл напрямую из UploadButton
            outputs=[anomaly_plot, report_output],
        )
        
        gr.Markdown(
            """
            ---
            **Важно:** 
            - Система предназначена для вспомогательной диагностики
            - Результаты не заменяют консультацию специалиста
            - При обнаружении аномалий рекомендуется обратиться к врачу
            """
        )
    
    return interface


if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7861, max_attempts=10):
        """Найти свободный порт начиная с start_port."""
        for i in range(max_attempts):
            port = start_port + i
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"Не удалось найти свободный порт в диапазоне {start_port}-{start_port + max_attempts - 1}")
    
    interface = create_medical_interface()
    
    # Находим свободный порт
    port = find_free_port(7861)
    logger.info(f"Запуск сервера на порту {port}...")
    
    try:
        interface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=port,
            show_error=True,
            quiet=False,
        )
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        raise

