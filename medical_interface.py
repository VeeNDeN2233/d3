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


def analyze_baby_video(video_path) -> Tuple[Optional[str], Optional[str]]:
    """
    Основная функция анализа видео.
    
    Args:
        video_path: Путь к видео файлу (может быть str или dict от Gradio)
    
    Returns:
        Tuple (anomaly_plot_path, report_json)
    """
    global _model, _detector, _config, _video_processor, _pose_processor
    
    if _model is None or _detector is None:
        return None, "Ошибка: Модели не загружены! Нажмите 'Загрузить модели'."
    
    try:
        # gr.File возвращает объект с атрибутом .name или строку
        if video_path is None:
            return None, "Ошибка: Видео не загружено!"
        
        # video_path теперь приходит из текстового поля (строка с путем)
        logger.info(f"Получен video_path типа: {type(video_path)}, значение: {video_path}")
        
        # Это должна быть строка с путем к файлу
        actual_path = str(video_path).strip() if video_path else ""
        
        if not actual_path:
            return None, "Ошибка: Путь к видео пустой!"
        
        video_path_obj = Path(actual_path)
        
        # Проверяем существование файла
        if not video_path_obj.exists():
            # Пробуем найти файл в других местах
            filename = video_path_obj.name
            possible_locations = [
                video_path_obj,
                Path("test_videos") / filename,
                Path("uploads") / filename,
                Path(filename),  # Текущая директория
            ]
            
            found = False
            for loc in possible_locations:
                if loc.exists():
                    video_path_obj = loc
                    found = True
                    break
            
            if not found:
                return None, f"Ошибка: Файл не найден: {actual_path}\nПопробуйте загрузить видео снова."
        
        # Обработка видео
        keypoints_list, errors, is_anomaly = process_video(
            video_path_obj, _video_processor, _pose_processor, _detector, _config
        )
        
        # Создаем временную директорию для результатов
        output_dir = Path("results") / f"analysis_{video_path_obj.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Визуализация
        visualize_results(errors, is_anomaly, output_dir, video_path_obj.stem, _detector.threshold)
        
        # Генерация медицинского отчета
        report = generate_medical_report(
            video_path_obj, errors, is_anomaly, _detector, output_dir
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
        return None, f"Ошибка анализа: {str(e)}\n\nДетали: {type(e).__name__}: {str(e)}"


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
    
    with gr.Blocks(title="Детектор аномалий движений младенцев") as interface:
        gr.Markdown(
            """
            # 🍼 Детектор аномалий движений младенцев
            
            **Используется улучшенная модель: Bidirectional LSTM + Attention**
            
            Загрузите видео младенца для анализа движений и оценки риска нарушений моторики.
            
            **Инструкция:**
            1. Нажмите "Загрузить модели" для инициализации системы
            2. Загрузите видео файл (MP4, AVI, MOV, MKV)
            3. Нажмите "Анализировать видео"
            4. Просмотрите результаты и медицинский отчет
            """
        )
        
        with gr.Row():
            with gr.Column():
                model_status = gr.Textbox(
                    label="Статус моделей",
                    value="Нажмите 'Загрузить модели' для инициализации",
                    interactive=False,
                )
                load_models_btn = gr.Button("Загрузить модели", variant="primary")
            
            with gr.Column():
                video_input = gr.UploadButton(
                    label="Загрузите видео младенца (MP4, AVI, MOV, MKV)",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                    file_count="single",
                )
                video_path_display = gr.Textbox(
                    label="Путь к загруженному файлу",
                    interactive=False,
                    visible=False,
                )
                analyze_btn = gr.Button("Анализировать видео", variant="primary")
        
        with gr.Row():
            with gr.Column():
                anomaly_plot = gr.Image(label="График аномалий")
            
            with gr.Column():
                report_output = gr.Textbox(
                    label="Медицинский отчет",
                    lines=20,
                    max_lines=30,
                )
        
        # Обработчики событий
        load_models_btn.click(
            fn=load_models,
            outputs=model_status,
        )
        
        # Обработчик загрузки файла
        def handle_file_upload(files):
            if files is None or len(files) == 0:
                return None, "Файл не загружен"
            file_path = files[0].name if hasattr(files[0], 'name') else str(files[0])
            return file_path, f"Файл загружен: {Path(file_path).name}"
        
        video_input.upload(
            fn=handle_file_upload,
            inputs=video_input,
            outputs=[video_path_display, model_status],
        )
        
        analyze_btn.click(
            fn=analyze_baby_video,
            inputs=video_path_display,  # Используем путь из текстового поля
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
    interface = create_medical_interface()
    try:
        interface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7861,
            show_error=True,
            quiet=False,
        )
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        # Пробуем альтернативный порт
        logger.info("Пробуем порт 7862...")
        interface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7862,
            show_error=True,
            quiet=False,
        )

