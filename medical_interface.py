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
from utils.video_visualizer import create_skeleton_video_from_processed
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


def analyze_baby_video(video_file, age_weeks=None, gestational_age_weeks=None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Основная функция анализа видео.
    
    Args:
        video_file: Файл от Gradio File компонента
    
    Returns:
        Tuple (anomaly_plot_path, report_json)
    """
    global _model, _detector, _config, _video_processor, _pose_processor
    
    if _model is None or _detector is None:
        return None, None, "❌ Ошибка: Модели не загружены!\n\nНажмите 'Загрузить модели' для инициализации системы."
    
    try:
        if video_file is None:
            return None, None, "❌ Ошибка: Видео не загружено!\n\nПожалуйста, загрузите видео файл перед анализом."
        
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
                return None, None, "❌ Ошибка: Список файлов пуст!"
        
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
            return None, None, "❌ Ошибка: Не удалось определить путь к файлу!\n\nПопробуйте загрузить видео снова."
        
        # Нормализуем путь (исправляем обратные слэши на Windows)
        actual_path = Path(actual_path).resolve()
        logger.info(f"Обработка файла: {actual_path}")
        
        # Проверяем существование файла
        if not actual_path.exists():
            logger.error(f"Файл не существует: {actual_path}")
            return None, None, f"❌ Ошибка: Файл не найден!\n\nПуть: {actual_path}\n\nПопробуйте загрузить видео снова."
        
        if not actual_path.is_file():
            logger.error(f"Путь не является файлом: {actual_path}")
            return None, None, f"❌ Ошибка: Указанный путь не является файлом!\n\nПуть: {actual_path}"
        
        # Обработка видео
        # Сохраняем путь к исходному видео для создания видео с скелетом
        original_video_path = actual_path
        
        keypoints_list, errors, is_anomaly, sequences_array = process_video(
            actual_path, _video_processor, _pose_processor, _detector, _config
        )
        
        # Создаем временную директорию для результатов
        output_dir = Path("results") / f"analysis_{actual_path.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Визуализация
        visualize_results(errors, is_anomaly, output_dir, actual_path.stem, _detector.threshold)
        
        # Создаем видео с наложенным скелетом
        skeleton_video_path = output_dir / "video_with_skeleton.mp4"
        try:
            # Проверяем, что keypoints_list не пустой
            if not keypoints_list or len(keypoints_list) == 0:
                logger.warning("keypoints_list пуст, невозможно создать видео с скелетом")
                skeleton_video_path = None
            else:
                # Используем keypoints_list для создания видео с скелетом
                logger.info(f"Создание видео с скелетом из {len(keypoints_list)} кадров с keypoints")
                create_skeleton_video_from_processed(
                    original_video_path,
                    keypoints_list,
                    skeleton_video_path,
                    errors=errors,
                    is_anomaly=is_anomaly,
                    threshold=_detector.threshold
                )
                logger.info(f"Видео с скелетом создано: {skeleton_video_path}")
        except Exception as e:
            logger.error(f"Не удалось создать видео с скелетом: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            skeleton_video_path = None
        
        # Генерация медицинского отчета с детальным анализом
        report = generate_medical_report(
            actual_path, errors, is_anomaly, _detector, output_dir,
            age_weeks=age_weeks, gestational_age_weeks=gestational_age_weeks,
            sequences_array=sequences_array
        )
        
        # Пути к результатам
        plot_path = output_dir / "reconstruction_error.png"
        
        # Форматирование отчета для отображения
        report_text = format_medical_report(report)
        
        # Возвращаем путь к графику, видео с скелетом и отчет
        # Для Gradio Video нужно использовать абсолютный путь
        video_path_for_gradio = None
        if skeleton_video_path and skeleton_video_path.exists():
            # Проверяем размер файла
            file_size = skeleton_video_path.stat().st_size
            if file_size > 0:
                video_path_for_gradio = str(skeleton_video_path.resolve())
                logger.info(f"✅ Видео готово для отображения: {video_path_for_gradio} ({file_size / 1024 / 1024:.2f} MB)")
            else:
                logger.error(f"❌ Видео файл пуст: {skeleton_video_path}")
        else:
            logger.warning(f"❌ Видео не создано или не существует: {skeleton_video_path}")
        
        # Если видео не создано, возвращаем исходное видео как fallback
        if video_path_for_gradio is None:
            logger.warning("Видео с скелетом недоступно, используем исходное видео")
            if original_video_path.exists():
                video_path_for_gradio = str(original_video_path.resolve())
                logger.info(f"Используется исходное видео: {video_path_for_gradio}")
        
        return (
            str(plot_path.resolve()) if plot_path.exists() else None,
            video_path_for_gradio,
            report_text
        )
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        error_msg = f"❌ Ошибка при анализе видео:\n\n{str(e)}\n\n"
        error_msg += f"Тип ошибки: {type(e).__name__}\n\n"
        error_msg += "Пожалуйста, проверьте:\n"
        error_msg += "1. Видео файл загружен корректно\n"
        error_msg += "2. Модели загружены (нажмите 'Загрузить модели')\n"
        error_msg += "3. Формат видео поддерживается\n"
        return None, None, error_msg


def format_medical_report(report: Dict) -> str:
    """Форматировать медицинский отчет в формате GMA для отображения."""
    if not report:
        return "Ошибка: Отчет пуст"
    
    lines = []
    lines.append("=" * 70)
    lines.append("ОТЧЕТ ПО ОЦЕНКЕ ОБЩИХ ДВИЖЕНИЙ (GMA)")
    lines.append("=" * 70)
    lines.append("")
    
    # GMA оценка
    gma = report.get("gma_assessment", {})
    if gma:
        risk_level = gma.get("risk_level", "unknown").upper()
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
        
        lines.append("РЕЗУЛЬТАТ GMA ОЦЕНКИ:")
        lines.append(f"  {risk_emoji.get(risk_level, '⚪')} Риск двигательных нарушений: {risk_level}")
        lines.append(f"  Оценка общих движений: {gma.get('assessment_result', 'N/A')}")
        lines.append(f"  Риск ДЦП: {gma.get('cp_risk', 'N/A')}")
        
        lines.append("")
    else:
        # Fallback для старых отчетов
        anomaly = report.get("anomaly_detection", {})
        risk_level = anomaly.get("risk_level", "unknown").upper()
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
        lines.append("РЕЗУЛЬТАТ ОЦЕНКИ:")
        lines.append(f"  {risk_emoji.get(risk_level, '⚪')} Риск двигательных нарушений: {risk_level}")
        lines.append("")
    
    # Информация о пациенте
    patient_info = report.get("patient_info", {})
    if patient_info:
        lines.append("ДАННЫЕ ПАЦИЕНТА:")
        if "age_weeks" in patient_info:
            lines.append(f"  Возраст: {patient_info['age_weeks']:.0f} недель после родов")
        if "period" in patient_info:
            lines.append(f"  Период: {patient_info['period']}")
        if patient_info.get("premature"):
            lines.append(f"  Недоношенность: {patient_info.get('gestational_age_weeks', 'N/A')} недель")
            if "corrected_age" in patient_info and patient_info["corrected_age"]:
                lines.append(f"  Скорректированный возраст: {patient_info['corrected_age']:.0f} недель")
        lines.append("")
    
    # Статистика анализа
    stats = report.get("statistics", {})
    lines.append("СТАТИСТИКА АНАЛИЗА:")
    lines.append(f"  Проанализировано последовательностей: {stats.get('total_sequences', 'N/A')}")
    lines.append(f"  Аномальных последовательностей: {stats.get('anomalous_sequences', 'N/A')} ({stats.get('anomaly_rate', 0):.2f}%)")
    lines.append("")
    
    # Шкала тяжести
    detailed_analysis = report.get("detailed_analysis", {})
    severity_score = detailed_analysis.get("severity_score", {})
    if severity_score:
        severity_level = severity_score.get("severity_level", "")
        severity_color = severity_score.get("color", "gray")
        total_score = severity_score.get("total_score", 0)
        
        lines.append("ОБЩАЯ ОЦЕНКА ТЯЖЕСТИ:")
        if severity_color == "red":
            lines.append(f"  🔴 {severity_level} (балл: {total_score})")
        elif severity_color == "orange":
            lines.append(f"  🟡 {severity_level} (балл: {total_score})")
        else:
            lines.append(f"  🟢 {severity_level} (балл: {total_score})")
        lines.append("")
    
    # Детальный анализ аномалий
    # Показываем анализ, если есть аномалии ИЛИ есть аномалии амплитуды (критическое снижение)
    amplitude_analysis = detailed_analysis.get("amplitude_analysis", {})
    has_amplitude_anomalies = amplitude_analysis.get("has_amplitude_anomalies", False)
    
    if detailed_analysis.get("has_anomalies", False) or has_amplitude_anomalies:
        lines.append("ДЕТАЛЬНЫЙ АНАЛИЗ АНОМАЛИЙ:")
        lines.append(f"  Источник нормальных значений: {detailed_analysis.get('normal_statistics_source', 'N/A')}")
        lines.append("")
        
        # Асимметрия
        asymmetry = detailed_analysis.get("asymmetry", {})
        if asymmetry.get("has_asymmetry", False):
            lines.append("  🔍 Асимметрия движений:")
            for finding in asymmetry.get("findings", []):
                severity_icon = "🔴" if finding.get("severity") == "high" else "🟡"
                confidence = finding.get("confidence", "")
                lines.append(f"    {severity_icon} {finding['description']}")
                if "data" in finding:
                    data = finding["data"]
                    if "deviation_sigma" in data:
                        lines.append(f"      Отклонение: {data['deviation_sigma']:.2f}σ от нормы")
                    if "ratio" in data:
                        lines.append(f"      Соотношение левая/правая: {data['ratio']:.2f} (норма: {data.get('normal_ratio', 1.0):.2f})")
                if confidence:
                    lines.append(f"      Уверенность: {confidence}")
            lines.append("")
        
        # Анализ конкретных суставов
        joint_analysis = detailed_analysis.get("joint_analysis", {})
        if joint_analysis.get("findings"):
            lines.append("  🔍 Отклонения в движениях суставов:")
            for finding in joint_analysis["findings"]:
                severity_icon = "🔴" if finding.get("severity") == "high" else "🟡"
                confidence = finding.get("confidence", "")
                
                if finding["type"] == "reduced_movement":
                    lines.append(f"    {severity_icon} {finding['description']}")
                    if "data" in finding:
                        data = finding["data"]
                        if "reduction_percent" in data:
                            lines.append(f"      Амплитуда снижена на {data['reduction_percent']:.1f}%")
                        if "deviation_sigma" in data:
                            lines.append(f"      Отклонение: {data['deviation_sigma']:.2f}σ от нормы")
                    if confidence:
                        lines.append(f"      Уверенность: {confidence}")
                elif finding["type"] == "high_speed":
                    lines.append(f"    {severity_icon} {finding['description']}")
                    if "data" in finding:
                        data = finding["data"]
                        if "ratio" in data:
                            lines.append(f"      Скорость выше нормы в {data['ratio']:.1f} раз")
                        if "deviation_sigma" in data:
                            lines.append(f"      Отклонение: {data['deviation_sigma']:.2f}σ от нормы")
                    if confidence:
                        lines.append(f"      Уверенность: {confidence}")
            lines.append("")
        
        # Скорость движений
        speed_analysis = detailed_analysis.get("speed_analysis", {})
        if speed_analysis.get("has_speed_anomalies", False):
            lines.append("  🔍 Аномалии скорости движений:")
            for finding in speed_analysis.get("findings", []):
                severity_icon = "🔴" if finding.get("severity") == "high" else "🟡"
                confidence = finding.get("confidence", "")
                lines.append(f"    {severity_icon} {finding['description']}")
                if "data" in finding:
                    data = finding["data"]
                    if "deviation_sigma" in data:
                        lines.append(f"      Отклонение: {data['deviation_sigma']:.2f}σ от нормы")
                if confidence:
                    lines.append(f"      Уверенность: {confidence}")
            lines.append("")
        
        # Амплитуда движений
        amplitude_analysis = detailed_analysis.get("amplitude_analysis", {})
        if amplitude_analysis.get("has_amplitude_anomalies", False):
            lines.append("  🔍 Аномалии амплитуды движений:")
            for finding in amplitude_analysis.get("findings", []):
                severity_icon = "🔴" if finding.get("severity") == "high" else "🟡"
                confidence = finding.get("confidence", "")
                lines.append(f"    {severity_icon} {finding['description']}")
                if "data" in finding:
                    data = finding["data"]
                    if "reduction_percent" in data:
                        lines.append(f"      Снижение на {data['reduction_percent']:.1f}%")
                    if "deviation_sigma" in data:
                        lines.append(f"      Отклонение: {data['deviation_sigma']:.2f}σ от нормы")
                if confidence:
                    lines.append(f"      Уверенность: {confidence}")
            lines.append("")
    
    # Выявленные признаки (краткое резюме)
    detected_signs = gma.get("detected_signs", []) if gma else []
    if detected_signs:
        lines.append("ВЫЯВЛЕННЫЕ ПРИЗНАКИ (краткое резюме):")
        for i, sign in enumerate(detected_signs, 1):
            lines.append(f"  {i}. {sign}")
        lines.append("")
    
    # Детекция аномалий
    anomaly = report.get("anomaly_detection", {})
    if anomaly:
        lines.append("ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ:")
        lines.append(f"  Средний score аномалии: {anomaly.get('mean_anomaly_score', 'N/A'):.6f}" if isinstance(anomaly.get('mean_anomaly_score'), (int, float)) else f"  Средний score: {anomaly.get('mean_anomaly_score', 'N/A')}")
        lines.append(f"  Порог детекции: {anomaly.get('threshold', 'N/A'):.6f}" if isinstance(anomaly.get('threshold'), (int, float)) else f"  Порог: {anomaly.get('threshold', 'N/A')}")
        lines.append("")
    
    # Рекомендации
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append("РЕКОМЕНДАЦИИ:")
        for rec in recommendations:
            lines.append(f"  {rec}")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("")
    lines.append("⚠️ ВАЖНО: Данная система предназначена для вспомогательной диагностики.")
    lines.append("Результаты не заменяют консультацию специалиста по GMA.")
    lines.append("При выявлении высокого риска требуется консультация детского невролога.")
    
    return "\n".join(lines)


def create_medical_interface():
    """Создать Gradio интерфейс."""
    
    with gr.Blocks(title="GMA - Оценка общих движений") as interface:
        # Заголовок
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <h1 style="margin-bottom: 10px;">🍼 General Movements Assessment</h1>
                <p style="color: #666; font-size: 16px;">Автоматизированная система для раннего выявления риска двигательных нарушений</p>
            </div>
            """
        )
        
        # Основной контент в табах
        with gr.Tabs() as tabs:
            # Вкладка 1: Анализ
            with gr.Tab("📊 Анализ видео"):
                gr.Markdown("### Загрузка данных")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        video_input = gr.File(
                            label="Видео для анализа",
                            file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                            file_count="single",
                        )
                    with gr.Column(scale=1):
                        patient_age_weeks = gr.Number(
                            label="Возраст (недели)",
                            value=12,
                            minimum=0,
                            maximum=20,
                            step=1,
                        )
                        gestational_age = gr.Number(
                            label="Срок беременности (недели)",
                            value=40,
                            minimum=24,
                            maximum=42,
                            step=1,
                        )
                
                analyze_btn = gr.Button(
                    "🚀 Начать анализ",
                    variant="primary",
                    size="lg",
                    scale=1
                )
                
                gr.Markdown("---")
                
                # Результаты анализа
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📹 Видео с анализом")
                        skeleton_video = gr.Video(
                            label="Видео с наложенным скелетом",
                            height=400
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 График ошибки реконструкции")
                        anomaly_plot = gr.Image(
                            label="Динамика ошибки",
                            height=400
                        )
            
            # Вкладка 2: Отчет
            with gr.Tab("📄 Медицинский отчет"):
                report_output = gr.Textbox(
                    label="Результаты анализа",
                    lines=30,
                    max_lines=50,
                    interactive=False,
                )
            
            # Вкладка 3: Инструкции
            with gr.Tab("ℹ️ Инструкции"):
                gr.Markdown(
                    """
                    ### 📋 Инструкция по съемке видео для GMA
                    
                    **Условия съемки:**
                    - Ребенок лежит на спине, спокоен и внимателен
                    - Легко одет (без носков)
                    - Без сосок и игрушек
                    - Родители рядом, но не взаимодействуют с ребенком
                    - Съемка сверху, видны руки и ноги
                    - Длительность: 1-3 минуты
                    - Возраст: оптимально 12-14 недель после предполагаемой даты родов
                    
                    ---
                    
                    ### 🔬 Методология
                    
                    **Метод:** Анализ RGB-видео с использованием Bidirectional LSTM + Attention
                    
                    **Назначение:** Выявление ранних признаков церебрального паралича и других неврологических нарушений у младенцев (0-5 месяцев)
                    
                    **Точность:** Система обучена на данных MINI-RGBD (737 последовательностей здоровых младенцев)
                    
                    ---
                    
                    ### ⚠️ Важно
                    
                    - Система предназначена для **вспомогательной диагностики**
                    - Результаты **не заменяют** консультацию специалиста
                    - При обнаружении аномалий рекомендуется обратиться к врачу
                    """
                )
        
        # Индикатор статуса загрузки моделей
        with gr.Row():
            model_status = gr.Markdown(
                value="⏳ **Инициализация системы...** Загрузка моделей при старте.",
                visible=True,
            )
        
        # Автоматическая загрузка моделей при старте
        def load_models_and_update_status():
            """Загрузить модели и обновить статус."""
            status = load_models()
            # Форматируем статус для Markdown
            if "✅" in status:
                status_html = f"### ✅ **Система готова**\n\n{status.replace('✅ ', '')}"
            elif "❌" in status:
                status_html = f"### ❌ **Ошибка загрузки**\n\n{status.replace('❌ ', '')}"
            else:
                status_html = f"### ⏳ **{status}**"
            return status_html
        
        interface.load(
            fn=load_models_and_update_status,
            outputs=model_status,
        )
        
        # Обработчики событий
        analyze_btn.click(
            fn=analyze_baby_video,
            inputs=[video_input, patient_age_weeks, gestational_age],
            outputs=[anomaly_plot, skeleton_video, report_output],
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
            theme=gr.themes.Soft()
        )
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        raise

