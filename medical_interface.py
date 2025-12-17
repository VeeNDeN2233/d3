"""
Медицинский интерфейс для анализа видео младенцев.

Gradio интерфейс для загрузки видео и получения медицинского отчета.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import yaml

# Подавляем несущественные предупреждения asyncio на Windows
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")

# Импорт системы аутентификации
from auth.auth_manager import AuthManager

# Импорт новых модулей архитектуры
from core import (
    AppState,
    StateManager,
    AnalysisStep,
    AuthHandler,
    VideoProcessor as CoreVideoProcessor,
    StepManager,
    AnalysisPipeline,
)
from core.state_manager import AnalysisParameters
from utils.gradio_helpers import create_status_message, create_progress_html
from utils.analysis_cache import AnalysisCache
from utils.gradio_state_adapter import GradioStateAdapter
from utils.ui_state_manager import UIStateManager
from utils.dom_controller import get_dom_controller_js
import threading

# Импорт оптимизаций
from utils.model_cache import get_model_cache
from utils.performance_optimizer import (
    cache_result,
    optimize_memory,
    batch_process,
    get_performance_stats,
)

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

# Менеджер аутентификации (старый, для совместимости)
_auth_manager = AuthManager()

# Новые менеджеры архитектуры
_state_manager = StateManager()
_auth_handler = AuthHandler()
_core_video_processor = CoreVideoProcessor()
_analysis_pipeline: Optional[AnalysisPipeline] = None
_cancel_event: Optional[threading.Event] = None
_analysis_cache = AnalysisCache()
_gradio_state_adapter = GradioStateAdapter(_state_manager)
_step_manager = StepManager(_state_manager)
_ui_state_manager = UIStateManager(_state_manager)

# Флаг для lazy loading моделей
_models_loading = False
_model_loading_lock = threading.Lock()


def load_models_lazy(config_path: str = "config.yaml", checkpoint_path: str = "checkpoints/best_model_advanced.pt", force: bool = False):
    """
    Lazy loading моделей - загружает только при необходимости.
    
    Args:
        config_path: Путь к конфигурации
        checkpoint_path: Путь к checkpoint модели
        force: Принудительная перезагрузка
    
    Returns:
        Сообщение о статусе загрузки
    """
    global _model, _detector, _config, _video_processor, _pose_processor, _models_loading, _model_loading_lock
    global _state_manager
    
    # Проверяем, не загружается ли уже
    with _model_loading_lock:
        if _models_loading:
            return "Модели загружаются, пожалуйста, подождите..."
        
        if _model is not None and not force:
            _state_manager.update_models(is_loaded=True, status_message="Модели уже загружены")
            return "Модели уже загружены"
        
        _models_loading = True
    
    try:
        _state_manager.update_models(is_loaded=False, status_message="Загрузка моделей...")
        
        # Проверяем кэш моделей
        model_cache = get_model_cache()
        checkpoint = Path(checkpoint_path)
        
        cached = model_cache.get(checkpoint, "bidir_lstm")
        if cached is not None and not force:
            _model, _detector = cached
            logger.info("Модели загружены из кэша")
            _state_manager.update_models(is_loaded=True, status_message="Модели загружены из кэша")
        else:
            # Загружаем конфигурацию
            with open(config_path, "r", encoding="utf-8") as f:
                _config = yaml.safe_load(f)
            
            # Проверяем GPU
            device = model_cache.get_device()
            if device.type != "cuda":
                _state_manager.update_models(
                    is_loaded=False,
                    loading_error="GPU недоступен",
                    status_message="Ошибка: GPU недоступен!"
                )
                return "Ошибка: GPU недоступен!"
            
            # Загружаем модель и детектор
            _model, _detector = load_model_and_detector(checkpoint, _config, device, model_type="bidir_lstm")
            
            # Логируем информацию об устройстве
            logger.info(f"✅ Основная модель загружена на устройство: {device}")
            if device.type == "cuda":
                logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info("ℹ️  MediaPipe Pose работает на CPU (это нормально и быстро)")
            
            # Сохраняем в кэш
            model_cache.set(checkpoint, "bidir_lstm", _model, _detector)
        
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
        
        device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        status_msg = f"Модели загружены успешно. (Bidirectional LSTM + Attention)\nУстройство: {device_name}\nПорог: {_detector.threshold:.6f}\n\nПримечание: MediaPipe Pose работает на CPU (это нормально)"
        _state_manager.update_models(is_loaded=True, status_message=status_msg)
        
        return status_msg
        
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}", exc_info=True)
        error_msg = f"Ошибка загрузки: {str(e)}"
        _state_manager.update_models(
            is_loaded=False,
            loading_error=str(e),
            status_message=error_msg
        )
        return error_msg
    finally:
        with _model_loading_lock:
            _models_loading = False


@cache_result(max_size=1)
def load_models(config_path: str = "config.yaml", checkpoint_path: str = "checkpoints/best_model_advanced.pt"):
    """Загрузить модели один раз при старте с кэшированием (старая функция для совместимости)."""
    return load_models_lazy(config_path, checkpoint_path, force=False)


def analyze_baby_video(
    video_file,
    age_weeks=None,
    gestational_age_weeks=None,
    session_token_state=None,
    progress=gr.Progress()
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Основная функция анализа видео с поддержкой прогресса и кэширования.
    
    Args:
        video_file: Файл от Gradio File компонента
        age_weeks: Возраст ребенка в неделях
        gestational_age_weeks: Гестационный возраст
        session_token_state: Токен сессии
        progress: Объект прогресса Gradio
    
    Returns:
        Tuple (anomaly_plot_path, video_path, report_text)
    """
    global _model, _detector, _config, _video_processor, _pose_processor, _auth_manager
    global _core_video_processor, _analysis_cache, _cancel_event
    
    # Проверка аутентификации
    session_token = session_token_state if session_token_state else None
    auth_success, user_data, auth_message = _auth_manager.require_auth(session_token)
    
    if not auth_success:
        return None, None, f"Ошибка: {auth_message}\n\nПожалуйста, войдите в систему."
    
    if _model is None or _detector is None:
        return None, None, "Ошибка: Модели не загружены!\n\nСистема инициализируется, пожалуйста, подождите."
    
    try:
        if video_file is None:
            return None, None, "Ошибка: Видео не загружено!\n\nПожалуйста, загрузите видео файл перед анализом."
        
        # Используем новый VideoProcessor для обработки файла
        progress(0.05, desc="Обработка файла...")
        actual_path = _core_video_processor.get_video_path(video_file)
        
        if actual_path is None:
            return None, None, "Ошибка: Не удалось определить путь к файлу.\n\nПопробуйте загрузить видео снова."
        
        # Валидация файла
        is_valid, error_msg = _core_video_processor.validate_video(actual_path)
        if not is_valid:
            return None, None, f"Ошибка валидации видео: {error_msg}"
        
        # Проверка кэша
        progress(0.1, desc="Проверка кэша...")
        age_weeks = age_weeks or 12
        gestational_age_weeks = gestational_age_weeks or 40
        
        cached_results = _analysis_cache.get(actual_path, age_weeks, gestational_age_weeks)
        if cached_results:
            logger.info("Используются результаты из кэша")
            progress(1.0, desc="Результаты загружены из кэша")
            return (
                cached_results.get('plot_path'),
                cached_results.get('video_path'),
                cached_results.get('report_text')
            )
        
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
        
        # Проверка отмены перед генерацией отчета
        if _cancel_event and _cancel_event.is_set():
            _state_manager.update_analysis(is_cancelled=True, is_running=False)
            return None, None, "Анализ отменен пользователем"
        
        # Генерация медицинского отчета
        progress(0.8, desc="Генерация медицинского отчета...")
        _state_manager.update_analysis(progress=0.8, current_step="Генерация медицинского отчета")
        report = generate_medical_report(
            actual_path, errors, is_anomaly, _detector, output_dir,
            age_weeks=age_weeks, gestational_age_weeks=gestational_age_weeks,
            sequences_array=sequences_array
        )
        
        # Пути к результатам
        plot_path = output_dir / "reconstruction_error.png"
        
        # Форматирование отчета
        report_text = format_medical_report(report)
        
        # Сохраняем результаты в состояние
        _state_manager.update_analysis(
            results={
                'plot_path': str(plot_path.resolve()) if plot_path.exists() else None,
                'report_text': report_text,
            }
        )
        
        # Возвращаем путь к графику, видео с скелетом и отчет
        # Для Gradio Video нужно использовать абсолютный путь
        video_path_for_gradio = None
        if skeleton_video_path and isinstance(skeleton_video_path, Path):
            if skeleton_video_path.exists():
                # Проверяем размер файла
                try:
                    file_size = skeleton_video_path.stat().st_size
                    if file_size > 0:
                        # Используем абсолютный путь с нормализацией для Windows
                        abs_path = skeleton_video_path.resolve()
                        video_path_for_gradio = str(abs_path)
                        
                        # Проверяем, что файл действительно читается
                        try:
                            with open(abs_path, 'rb') as f:
                                f.read(1024)  # Читаем первые 1024 байта для проверки
                            logger.info(f"Видео готово для отображения: {video_path_for_gradio} ({file_size / 1024 / 1024:.2f} MB)")
                        except Exception as e:
                            logger.error(f"❌ Не удалось прочитать видео файл: {e}")
                            video_path_for_gradio = None
                    else:
                        logger.error(f"❌ Видео файл пуст: {skeleton_video_path}")
                except Exception as e:
                    logger.error(f"❌ Ошибка при проверке видео файла: {e}")
            else:
                logger.warning(f"❌ Видео не существует: {skeleton_video_path}")
        else:
            logger.warning(f"❌ Видео не создано или путь некорректен: {skeleton_video_path}")
        
        # Если видео не создано, возвращаем исходное видео как fallback
        if video_path_for_gradio is None:
            logger.warning("Видео с скелетом недоступно, используем исходное видео")
            if original_video_path.exists():
                try:
                    abs_path = original_video_path.resolve()
                    # Проверяем, что файл читается
                    with open(abs_path, 'rb') as f:
                        f.read(1024)
                    video_path_for_gradio = str(abs_path)
                    logger.info(f"Используется исходное видео: {video_path_for_gradio}")
                except Exception as e:
                    logger.error(f"Не удалось использовать исходное видео: {e}")
        
        # Очистка памяти после обработки
        progress(0.95, desc="Очистка памяти...")
        _state_manager.update_analysis(progress=0.95, current_step="Очистка памяти")
        optimize_memory()
        
        # Подготовка результатов для кэширования
        results = {
            'plot_path': str(plot_path.resolve()) if plot_path.exists() else None,
            'video_path': video_path_for_gradio,
            'report_text': report_text
        }
        
        # Сохранение в кэш
        _analysis_cache.set(actual_path, age_weeks, gestational_age_weeks, results)
        
        # Обновляем состояние анализа
        _state_manager.update_analysis(
            is_running=False,
            progress=1.0,
            current_step="Анализ завершен",
            results=results
        )
        
        progress(1.0, desc="Анализ завершен")
        
        return (
            results['plot_path'],
            results['video_path'],
            results['report_text']
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
    """Создать клинический процедурный интерфейс для анализа движений младенцев."""
    
    # Клинический CSS для медицинского интерфейса
    custom_css = """
    /* Скрытие страницы входа, когда header виден (пользователь авторизован) */
    body:has(.header-panel) .gr-column:has(button:contains("Войти в систему")),
    body:has(.header-panel) .gr-column:has(input[type="password"][placeholder*="пароль" i]),
    body:has(.header-panel) .gr-column:has(input[type="password"][placeholder*="password" i]) {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
    }
    
    /* Скрытие кнопки "Войти в систему", если header виден */
    body:has(.header-panel) button:contains("Войти в систему"),
    .header-panel ~ * button:contains("Войти в систему"),
    button:contains("Войти в систему"):has(+ .header-panel) {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
    }
    
    /* Более агрессивное скрытие - по наличию email в header */
    .header-panel:has(span:contains("@")) ~ * .gr-column:has(button:contains("Войти в систему")) {
        display: none !important;
    }
    
    /* Базовые стили */
    * {
        box-sizing: border-box !important;
    }
    
    .gradio-container {
        background: #f5f7fa !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        color: #2c3e50 !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Основной контент */
    .main {
        background: transparent !important;
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Контейнер для контента */
    .container {
        max-width: 1200px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 40px 24px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Форма входа - центрированная */
    .login-form-container {
        width: 100% !important;
        max-width: 420px !important;
        margin: 0 auto !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Группа формы */
    .gr-group {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 32px !important;
        margin: 0 0 24px 0 !important;
        display: flex !important;
        flex-direction: column !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Верхняя панель */
    .header-panel {
        background: #ffffff !important;
        border-bottom: 1px solid #e1e8ed !important;
        padding: 20px 24px !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        flex-shrink: 0 !important;
    }
    
    .header-panel > div {
        max-width: 1200px !important;
        width: 100% !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    /* Заголовки - клинический стиль */
    h1 {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 24px !important;
        letter-spacing: -0.3px !important;
        margin: 0 !important;
        line-height: 1.3 !important;
    }
    
    h2 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 20px !important;
        margin: 0 0 12px 0 !important;
        line-height: 1.4 !important;
    }
    
    h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        margin: 0 0 16px 0 !important;
        line-height: 1.4 !important;
    }
    
    p {
        margin: 0 0 16px 0 !important;
        line-height: 1.6 !important;
    }
    
    h4 {
        color: #4a5568 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin: 0 0 8px 0 !important;
    }
    
    /* Кнопки - клинический стиль */
    button.primary {
        background: #4a90e2 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 12px 32px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        transition: background 0.2s ease !important;
        color: white !important;
        cursor: pointer !important;
    }
    
    button.primary:hover:not(:disabled) {
        background: #357abd !important;
    }
    
    button.primary:disabled {
        background: #cbd5e0 !important;
        cursor: not-allowed !important;
        color: #a0aec0 !important;
    }
    
    button.secondary {
        background: #718096 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        color: white !important;
        transition: background 0.2s ease !important;
    }
    
    button.secondary:hover {
        background: #4a5568 !important;
    }
    
    button.stop {
        background: #e53e3e !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        color: white !important;
        transition: background 0.2s ease !important;
    }
    
    button.stop:hover {
        background: #c53030 !important;
    }
    
    /* Поля ввода - клинический стиль */
    input[type="text"], input[type="password"], input[type="number"], input[type="email"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e0 !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        transition: border-color 0.2s ease !important;
        color: #2d3748 !important;
        font-family: inherit !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        display: block !important;
        height: 42px !important;
        line-height: 1.5 !important;
    }
    
    /* Принудительно делаем textarea однострочными для полей входа */
    .gr-textbox textarea {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        resize: none !important;
        overflow: hidden !important;
        line-height: 22px !important;
        padding: 10px 14px !important;
        white-space: nowrap !important;
    }
    
    /* Скрываем полосы прокрутки и resize handle */
    .gr-textbox textarea::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    .gr-textbox textarea {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }
    
    /* Убираем resize handle */
    .gr-textbox textarea::-webkit-resizer {
        display: none !important;
    }
    
    /* Для полей входа и регистрации - строго однострочные */
    .gr-group .gr-textbox textarea {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        resize: none !important;
        overflow: hidden !important;
        line-height: 22px !important;
        padding: 10px 14px !important;
    }
    
    textarea {
        background: #ffffff !important;
        border: 1px solid #cbd5e0 !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        transition: border-color 0.2s ease !important;
        color: #2d3748 !important;
        font-family: inherit !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        resize: vertical !important;
        min-height: 42px !important;
        line-height: 1.5 !important;
    }
    
    /* Контейнеры для полей ввода */
    .gr-textbox,
    .gr-number {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        margin-bottom: 20px !important;
    }
    
    .gr-textbox:last-child,
    .gr-number:last-child {
        margin-bottom: 0 !important;
    }
    
    .gr-textbox label,
    .gr-number label {
        margin-bottom: 6px !important;
    }
    
    .gr-textbox input,
    .gr-textbox textarea,
    .gr-number input {
        margin: 0 !important;
    }
    
    input[type="text"]:focus, input[type="password"]:focus, input[type="number"]:focus, textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1) !important;
        outline: none !important;
    }
    
    input[type="text"]:disabled, input[type="number"]:disabled, textarea:disabled {
        background: #f7fafc !important;
        color: #a0aec0 !important;
        cursor: not-allowed !important;
    }
    
    /* Лейблы */
    label {
        font-weight: 500 !important;
        font-size: 14px !important;
        color: #4a5568 !important;
        margin-bottom: 8px !important;
        display: block !important;
        line-height: 1.5 !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* Убираем цветные фоны с лейблов Gradio */
    .gr-textbox > label,
    .gr-number > label,
    .gr-textbox label,
    .gr-number label {
        background: transparent !important;
        background-color: transparent !important;
        color: #4a5568 !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 0 !important;
    }
    
    /* Убираем все декоративные элементы с лейблов */
    label span,
    .gr-textbox label span {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Карточки */
    .card {
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin: 16px 0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        border: 1px solid #e8e8e8 !important;
    }
    
    /* Stepper / Wizard - пошаговый процесс */
    .stepper {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin: 0 0 32px 0 !important;
        padding: 24px 32px !important;
        background: #ffffff !important;
        border-bottom: 1px solid #e2e8f0 !important;
        position: relative !important;
        flex-shrink: 0 !important;
    }
    
    .stepper::before {
        content: '' !important;
        position: absolute !important;
        top: 20px !important;
        left: 24px !important;
        right: 24px !important;
        height: 2px !important;
        background: #e2e8f0 !important;
        z-index: 0 !important;
    }
    
    .step {
        position: relative !important;
        z-index: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        flex: 1 !important;
    }
    
    .step-circle {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #a0aec0 !important;
        margin-bottom: 8px !important;
    }
    
    .step.active .step-circle {
        background: #4a90e2 !important;
        border-color: #4a90e2 !important;
        color: #ffffff !important;
    }
    
    .step.completed .step-circle {
        background: #48bb78 !important;
        border-color: #48bb78 !important;
        color: #ffffff !important;
    }
    
    .step-label {
        font-size: 12px !important;
        color: #718096 !important;
        text-align: center !important;
        font-weight: 500 !important;
    }
    
    .step.active .step-label {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    /* Шаги процесса */
    .step-panel {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 32px !important;
        margin: 0 0 24px 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
    }
    
    .step-panel.disabled {
        opacity: 0.6 !important;
        pointer-events: none !important;
    }
    
    .step-panel.active {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1), 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }
    
    /* Группы элементов внутри панелей */
    .step-panel > * {
        margin-bottom: 24px !important;
        flex-shrink: 0 !important;
    }
    
    .step-panel > *:last-child {
        margin-bottom: 0 !important;
    }
    
    /* Группа формы входа */
    .gr-group {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 32px !important;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    .gr-group > * {
        margin-bottom: 20px !important;
    }
    
    .gr-group > *:last-child {
        margin-bottom: 0 !important;
    }
    
    /* Row и Column на flexbox */
    .gr-row {
        display: flex !important;
        flex-wrap: wrap !important;
        margin: 0 -12px 24px -12px !important;
        width: calc(100% + 24px) !important;
    }
    
    .gr-row:last-child {
        margin-bottom: 0 !important;
    }
    
    .gr-column {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        padding: 0 12px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    .gr-column[scale="1"] {
        flex: 0 0 auto !important;
    }
    
    .gr-column[scale="2"] {
        flex: 2 1 0 !important;
    }
    
    .gr-column[scale="3"] {
        flex: 3 1 0 !important;
    }
    
    /* Файловый загрузчик - клинический стиль */
    .file-upload {
        background: #f7fafc !important;
        border: 2px dashed #cbd5e0 !important;
        border-radius: 8px !important;
        padding: 64px 32px !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
        min-height: 240px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 24px 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    .file-upload:hover {
        border-color: #4a90e2 !important;
        background: #edf2f7 !important;
    }
    
    .file-upload.has-file {
        border-color: #48bb78 !important;
        background: #f0fff4 !important;
        border-style: solid !important;
    }
    
    /* Видео и изображения */
    video, img {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background: #f7fafc !important;
        margin: 0 !important;
        display: block !important;
        max-width: 100% !important;
        height: auto !important;
    }
    
    /* Контейнеры для видео и изображений */
    .gr-video,
    .gr-image {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        margin-bottom: 24px !important;
    }
    
    .gr-video:last-child,
    .gr-image:last-child {
        margin-bottom: 0 !important;
    }
    
    .gr-video video,
    .gr-image img {
        width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }
    
    /* Разделители */
    hr {
        border: none !important;
        border-top: 1px solid #e2e8f0 !important;
        margin: 32px 0 !important;
    }
    
    /* Кнопки */
    button {
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
    }
    
    .gr-button {
        margin: 16px 0 !important;
        display: flex !important;
        width: auto !important;
    }
    
    /* Textbox и другие компоненты */
    textarea {
        min-height: 120px !important;
        resize: vertical !important;
    }
    
    /* Группы компонентов */
    .gr-group {
        display: flex !important;
        flex-direction: column !important;
        margin-bottom: 24px !important;
        padding: 0 !important;
        width: 100% !important;
    }
    
    .gr-group:last-child {
        margin-bottom: 0 !important;
    }
    
    /* Статусные сообщения */
    .status-info {
        background: #ebf8ff !important;
        border-left: 4px solid #4a90e2 !important;
        padding: 16px 20px !important;
        border-radius: 6px !important;
        margin: 24px 0 !important;
        font-size: 14px !important;
        color: #2c5282 !important;
        line-height: 1.6 !important;
    }
    
    .status-success {
        background: #f0fff4 !important;
        border-left: 4px solid #48bb78 !important;
        padding: 16px 20px !important;
        border-radius: 6px !important;
        margin: 24px 0 !important;
        font-size: 14px !important;
        color: #22543d !important;
        line-height: 1.6 !important;
    }
    
    .status-error {
        background: #fff5f5 !important;
        border-left: 4px solid #e53e3e !important;
        padding: 16px 20px !important;
        border-radius: 6px !important;
        margin: 24px 0 !important;
        font-size: 14px !important;
        color: #742a2a !important;
        line-height: 1.6 !important;
    }
    
    .status-warning {
        background: #fffbeb !important;
        border-left: 4px solid #ed8936 !important;
        padding: 16px 20px !important;
        border-radius: 6px !important;
        margin: 24px 0 !important;
        font-size: 14px !important;
        color: #7c2d12 !important;
        line-height: 1.6 !important;
    }
    
    /* Placeholder для пустых состояний */
    .empty-state {
        text-align: center !important;
        padding: 48px 24px !important;
        color: #718096 !important;
        font-size: 14px !important;
    }
    
    .empty-state-title {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
        margin-bottom: 8px !important;
    }
    
    /* Информация о пользователе */
    .user-info {
        background: #f7fafc !important;
        border-radius: 4px !important;
        padding: 12px 16px !important;
        border: 1px solid #e2e8f0 !important;
        font-size: 13px !important;
        color: #4a5568 !important;
    }
    
    /* Результаты анализа */
    .results-panel {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 32px !important;
        margin: 32px 0 !important;
    }
    
    /* Markdown блоки */
    .gr-markdown {
        margin: 0 0 24px 0 !important;
        display: block !important;
        width: 100% !important;
    }
    
    .gr-markdown:last-child {
        margin-bottom: 0 !important;
    }
    
    .gr-markdown p {
        margin-bottom: 12px !important;
        line-height: 1.6 !important;
    }
    
    .gr-markdown p:last-child {
        margin-bottom: 0 !important;
    }
    
    .gr-markdown ul,
    .gr-markdown ol {
        margin: 8px 0 !important;
        padding-left: 24px !important;
    }
    
    .gr-markdown li {
        margin-bottom: 4px !important;
        line-height: 1.6 !important;
    }
    
    /* Адаптивность */
    @media (max-width: 768px) {
        .container {
            padding: 24px 16px !important;
        }
        
        .header-panel {
            padding: 16px !important;
        }
        
        .header-panel > div {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 12px !important;
        }
        
        .stepper {
            flex-direction: column !important;
            align-items: flex-start !important;
            padding: 20px 16px !important;
        }
        
        .stepper::before {
            display: none !important;
        }
        
        .step {
            flex-direction: row !important;
            width: 100% !important;
            margin-bottom: 16px !important;
            align-items: center !important;
        }
        
        .step-circle {
            margin-right: 12px !important;
            margin-bottom: 0 !important;
        }
        
        .step-panel {
            padding: 24px 16px !important;
        }
        
        .gr-row {
            flex-direction: column !important;
            margin: 0 0 20px 0 !important;
            width: 100% !important;
        }
        
        .gr-column {
            width: 100% !important;
            padding: 0 !important;
            margin-bottom: 16px !important;
        }
        
        .gr-column:last-child {
            margin-bottom: 0 !important;
        }
        
        .file-upload {
            padding: 40px 20px !important;
            min-height: 180px !important;
        }
    }
    """
    
    # JavaScript для скрытия страницы входа и оборачивания полей в формы
    js_hide_login_if_authenticated = """
    <script>
    (function() {
        // Функция для оборачивания полей пароля в формы
        function wrapPasswordFieldsInForms() {
            // Находим все группы с полями пароля
            const groups = document.querySelectorAll('.gr-group');
            groups.forEach((group, index) => {
                const passwordField = group.querySelector('input[type="password"]');
                if (passwordField && !passwordField.closest('form') && !passwordField.hasAttribute('form')) {
                    // Создаем форму с уникальным id
                    const formId = 'auth-form-' + index;
                    let form = document.getElementById(formId);
                    
                    if (!form) {
                        form = document.createElement('form');
                        form.id = formId;
                        form.setAttribute('onsubmit', 'return false;');
                        form.setAttribute('method', 'post');
                        form.setAttribute('autocomplete', 'on');
                        form.style.position = 'relative';
                        form.style.display = 'block';
                        
                        // Вставляем форму как обертку группы
                        const parent = group.parentNode;
                        parent.insertBefore(form, group);
                        form.appendChild(group);
                    }
                    
                    // Добавляем атрибут form ко всем полям в группе
                    const allInputs = group.querySelectorAll('input, button');
                    allInputs.forEach(field => {
                        if (!field.hasAttribute('form')) {
                            field.setAttribute('form', formId);
                        }
                    });
                }
            });
        }
        
        function hideLoginPage() {
            // Проверяем, есть ли header с email (признак авторизации)
            const headerPanel = document.querySelector('.header-panel');
            const hasHeader = headerPanel && headerPanel.offsetParent !== null && 
                            headerPanel.textContent.includes('@');
            
            if (hasHeader) {
                // Header виден - пользователь авторизован, скрываем страницу входа
                // Ищем все элементы с текстом "Войти в систему"
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {
                    if (el.textContent && el.textContent.includes('Войти в систему')) {
                        // Находим родительский Column (страницу входа)
                        let column = el.closest('.gr-column');
                        if (column) {
                            // Проверяем, что это действительно страница входа
                            const hasLoginForm = column.querySelector('input[type="password"]') || 
                                              column.textContent.includes('Email') ||
                                              column.textContent.includes('Пароль');
                            if (hasLoginForm) {
                                column.style.display = 'none';
                                column.style.visibility = 'hidden';
                                column.style.opacity = '0';
                                column.style.height = '0';
                                column.style.overflow = 'hidden';
                                column.style.pointerEvents = 'none';
                                column.setAttribute('aria-hidden', 'true');
                                
                                // Также скрываем все кнопки внутри
                                const buttons = column.querySelectorAll('button');
                                buttons.forEach(btn => {
                                    if (btn.textContent.includes('Войти в систему')) {
                                        btn.style.display = 'none';
                                        btn.style.visibility = 'hidden';
                                        btn.style.opacity = '0';
                                        btn.setAttribute('disabled', 'true');
                                    }
                                });
                            }
                        }
                        // Также скрываем саму кнопку, если она найдена отдельно
                        if (el.tagName === 'BUTTON' && el.textContent.includes('Войти в систему')) {
                            el.style.display = 'none';
                            el.style.visibility = 'hidden';
                            el.style.opacity = '0';
                            el.style.height = '0';
                            el.style.width = '0';
                            el.style.padding = '0';
                            el.style.margin = '0';
                            el.style.pointerEvents = 'none';
                            el.setAttribute('disabled', 'true');
                        }
                    }
                });
            }
        }
        
        // Вызываем функции оборачивания полей в формы
        wrapPasswordFieldsInForms();
        setTimeout(wrapPasswordFieldsInForms, 100);
        setTimeout(wrapPasswordFieldsInForms, 500);
        
        // Вызываем сразу и повторно с задержками
        hideLoginPage();
        setTimeout(hideLoginPage, 100);
        setTimeout(hideLoginPage, 500);
        setTimeout(hideLoginPage, 1000);
        
        // Вызываем при загрузке страницы
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                wrapPasswordFieldsInForms();
                hideLoginPage();
            });
        }
        
        // Также проверяем после обновления DOM (для Gradio)
        const observer = new MutationObserver(function() {
            setTimeout(function() {
                wrapPasswordFieldsInForms();
                hideLoginPage();
            }, 50);
        });
        observer.observe(document.body, { childList: true, subtree: true, attributes: true });
        
        // Также слушаем события Gradio
        window.addEventListener('load', function() {
            wrapPasswordFieldsInForms();
            hideLoginPage();
        });
        document.addEventListener('DOMContentLoaded', function() {
            wrapPasswordFieldsInForms();
            hideLoginPage();
        });
        
        // Постоянная проверка каждые 100мс (на случай, если Gradio перерисовывает компоненты)
        setInterval(function() {
            const headerPanel = document.querySelector('.header-panel');
            const hasHeader = headerPanel && headerPanel.offsetParent !== null && 
                            headerPanel.textContent.includes('@');
            
            if (hasHeader) {
                // Находим и скрываем все кнопки "Войти в систему"
                const loginButtons = document.querySelectorAll('button');
                loginButtons.forEach(btn => {
                    if (btn.textContent && btn.textContent.includes('Войти в систему')) {
                        btn.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; width: 0 !important; padding: 0 !important; margin: 0 !important; pointer-events: none !important;';
                        btn.setAttribute('disabled', 'true');
                        btn.setAttribute('aria-hidden', 'true');
                    }
                });
                
                // Скрываем страницу входа
                const loginColumns = document.querySelectorAll('.gr-column');
                loginColumns.forEach(col => {
                    const hasLoginForm = col.querySelector('input[type="password"]') || 
                                      col.textContent.includes('Email') ||
                                      col.textContent.includes('Войти в систему');
                    if (hasLoginForm) {
                        col.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important; pointer-events: none !important;';
                        col.setAttribute('aria-hidden', 'true');
                    }
                });
            }
        }, 100);
    })();
    </script>
    """
    
    with gr.Blocks(title="GMA - Оценка общих движений") as interface:
        # Используем DOM Controller для агрессивного управления элементами
        dom_controller_js = get_dom_controller_js()
        gr.HTML(value=dom_controller_js, visible=False)
        
        # Старый JavaScript для совместимости
        gr.HTML(value=js_hide_login_if_authenticated, visible=False)
        
        # Верхняя панель - только на главной странице (БЕЗ кнопки входа!)
        # Header показывается только когда main_page видна и содержит только email + кнопку выхода
        header_info = gr.Markdown(
            value="",
            visible=False,
        )
        
        # Скрытая кнопка выхода (активируется через JavaScript из header)
        logout_btn = gr.Button("Выйти", variant="stop", visible=False, elem_id="header-logout-btn")
        
        # Хранилище состояния
        session_token_storage = gr.State(value=None)
        is_authenticated = gr.State(value=False)
        current_user_data = gr.State(value=None)
        
        # Проверяем начальное состояние авторизации ПЕРЕД созданием страниц
        # Используем UIStateManager для определения видимости
        show_login, show_register, show_main = _ui_state_manager.get_page_visibility()
        logger.info(f"Начальное состояние UI: show_login={show_login}, show_register={show_register}, show_main={show_main}")
        
        # СТРАНИЦА 1: ВХОД В СИСТЕМУ (видимость определяется через UIStateManager)
        with gr.Column(visible=show_login) as login_page:
            gr.Markdown(
                """
                <div style="max-width: 420px; margin: 60px auto; padding: 0 24px;">
                """
            )
            
            # Заголовок
            gr.Markdown(
                """
                <div style="text-align: center; margin-bottom: 32px;">
                    <h2 style="color: #1a202c; margin: 0 0 8px 0; font-size: 24px; font-weight: 600; line-height: 1.3;">Доступ к системе анализа движений</h2>
                    <p style="color: #718096; font-size: 14px; margin: 0; line-height: 1.5;">Вход для зарегистрированных специалистов</p>
                </div>
                """
            )
            
            # Форма входа
            gr.HTML(value='<form id="login-form" onsubmit="return false;">', visible=False)
            with gr.Group():
                login_email = gr.Textbox(
                    label="Email",
                    placeholder="your.email@example.com",
                    container=True,
                    lines=1,
                    max_lines=1,
                )
                login_password = gr.Textbox(
                    label="Пароль",
                    type="password",
                    placeholder="Введите пароль",
                    container=True,
                    lines=1,
                    max_lines=1,
                )
                
                login_btn = gr.Button(
                    "Войти в систему",
                    variant="primary",
                    size="lg",
                )
                
                login_status = gr.Markdown(
                    visible=True,
                    value="",
                    elem_classes=["status-message"],
                )
            gr.HTML(value='</form>', visible=False)
            
            # Дополнительные ссылки
            with gr.Row():
                show_register_btn = gr.Button(
                    "Зарегистрироваться",
                    variant="secondary",
                    size="sm",
                    scale=0,
                )
            
            gr.Markdown(
                """
                <div style="text-align: center; margin-top: 16px;">
                    <p style="margin: 0; font-size: 13px; color: #718096;">
                        Нет учетной записи?
                    </p>
                </div>
                """
            )
            
            gr.Markdown("</div></div>")  # Закрываем контейнеры
            
            # Скрытая форма регистрации (отдельный экран)
            with gr.Column(visible=False) as register_page:
                gr.Markdown(
                    """
                    <div style="max-width: 420px; margin: 60px auto; padding: 0 24px;">
                    """
                )
                
                # Заголовок регистрации
                gr.Markdown(
                    """
                    <div style="text-align: center; margin-bottom: 32px;">
                        <h2 style="color: #1a202c; margin: 0 0 8px 0; font-size: 24px; font-weight: 600; line-height: 1.3;">Регистрация в системе</h2>
                        <p style="color: #718096; font-size: 14px; margin: 0; line-height: 1.5;">Создание учетной записи для доступа к системе анализа</p>
                    </div>
                    """
                )
                
                # Форма регистрации
                gr.HTML(value='<form id="register-form" onsubmit="return false;">', visible=False)
                with gr.Group():
                    reg_email = gr.Textbox(
                        label="Email",
                        placeholder="your.email@example.com",
                        container=True,
                        lines=1,
                        max_lines=1,
                    )
                    reg_full_name = gr.Textbox(
                        label="Полное имя",
                        placeholder="Иван Иванов",
                        container=True,
                        lines=1,
                        max_lines=1,
                    )
                    reg_password = gr.Textbox(
                        label="Пароль",
                        type="password",
                        placeholder="Минимум 6 символов",
                        container=True,
                        lines=1,
                        max_lines=1,
                    )
                    reg_password_confirm = gr.Textbox(
                        label="Подтверждение пароля",
                        type="password",
                        placeholder="Повторите пароль",
                        container=True,
                        lines=1,
                        max_lines=1,
                    )
                    
                    register_btn = gr.Button(
                        "Зарегистрироваться",
                        variant="primary",
                        size="lg",
                        scale=1,
                    )
                    
                    reg_status = gr.Markdown(
                        visible=True,
                        value="",
                        elem_classes=["status-message"],
                    )
                gr.HTML(value='</form>', visible=False)
                
                # Ссылка возврата
                with gr.Row():
                    show_login_btn = gr.Button(
                        "Войти",
                        variant="secondary",
                        size="sm",
                        scale=0,
                    )
                
                gr.Markdown(
                    """
                    <div style="text-align: center; margin-top: 16px;">
                        <p style="margin: 0; font-size: 13px; color: #718096;">
                            Уже есть учетная запись?
                        </p>
                    </div>
                    """
                )
                
                gr.Markdown("</div></div>")  # Закрываем контейнеры
        
        # СТРАНИЦА 2: ГЛАВНАЯ СТРАНИЦА С ФУНКЦИЯМИ (видимость определяется через UIStateManager)
        with gr.Column(visible=show_main) as main_page:
            gr.Markdown(
                """
                <div class="container">
                """
            )
            
            # Состояния для управления шагами (синхронизируются с StateManager через StepManager)
            def get_current_step():
                state = _state_manager.get_state()
                step_mapping = {
                    AnalysisStep.UPLOAD: 1,
                    AnalysisStep.PARAMETERS: 2,
                    AnalysisStep.ANALYSIS: 3,
                    AnalysisStep.RESULTS: 4,
                }
                return step_mapping.get(state.current_step, 1)
            def get_video_uploaded():
                return _state_manager.get_state().video.is_uploaded
            
            current_step = gr.State(value=get_current_step)
            video_uploaded = gr.State(value=get_video_uploaded)
            
            # Stepper - индикатор шагов
            stepper_html = gr.Markdown(
                value="""
                <div class="stepper">
                    <div class="step active" id="step-1">
                        <div class="step-circle">1</div>
                        <div class="step-label">Загрузка видео</div>
                    </div>
                    <div class="step" id="step-2">
                        <div class="step-circle">2</div>
                        <div class="step-label">Параметры</div>
                    </div>
                    <div class="step" id="step-3">
                        <div class="step-circle">3</div>
                        <div class="step-label">Анализ</div>
                    </div>
                    <div class="step" id="step-4">
                        <div class="step-circle">4</div>
                        <div class="step-label">Результаты</div>
                    </div>
                </div>
                """
            )
            
            # ШАГ 1: Загрузка видео
            with gr.Group(visible=True, elem_classes=["step-panel", "active"]) as step1_panel:
                gr.Markdown(
                    """
                    <h2 style="margin-bottom: 12px;">Шаг 1: Загрузка видео</h2>
                    <p style="color: #718096; font-size: 14px; margin-bottom: 24px; line-height: 1.6;">
                        Загрузите видео младенца для анализа движений. Видео должно соответствовать требованиям съемки для GMA.
                    </p>
                    """
                )
                
                video_input = gr.File(
                    label="Видео для анализа",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                    file_count="single",
                    height=200,
                )
                
                gr.Markdown(
                    """
                    <div class="status-info" style="margin-top: 24px;">
                        <strong style="display: block; margin-bottom: 8px;">Требования к видео:</strong>
                        <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                            <li>Формат: MP4, AVI, MOV, MKV, WebM</li>
                            <li>Длительность: 1-3 минуты</li>
                            <li>Положение камеры: сверху, видны руки и ноги</li>
                            <li>Ребенок: лежит на спине, спокоен, легко одет</li>
                        </ul>
                    </div>
                    """
                )
                
                video_status = gr.Markdown(
                    value="<div class='empty-state'><div class='empty-state-title'>Видео не загружено</div><p>Перетащите файл в область выше или нажмите для выбора</p></div>",
                    visible=True
                )
                
                # Кнопка перехода к следующему шагу
                next_to_step2_btn = gr.Button(
                    "Далее",
                    variant="primary",
                    size="lg",
                    interactive=False,
                )
            
            # ШАГ 2: Клинические параметры
            with gr.Group(visible=False, elem_classes=["step-panel", "disabled"]) as step2_panel:
                gr.Markdown(
                    """
                    <h2 style="margin-bottom: 12px;">Шаг 2: Клинические параметры</h2>
                    <p style="color: #718096; font-size: 14px; margin-bottom: 24px; line-height: 1.6;">
                        Укажите параметры для корректировки модели анализа. Эти данные используются для адаптации алгоритма под возраст ребенка.
                    </p>
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        patient_age_weeks = gr.Number(
                            label="Возраст ребенка (недели)",
                            value=12,
                            minimum=0,
                            maximum=20,
                            step=1,
                            container=True,
                        )
                    with gr.Column(scale=1):
                        gestational_age = gr.Number(
                            label="Гестационный возраст (недели)",
                            value=40,
                            minimum=24,
                            maximum=42,
                            step=1,
                            container=True,
                        )
                
                # Кнопки навигации
                with gr.Row():
                    back_to_step1_btn = gr.Button(
                        "Назад",
                        variant="secondary",
                        scale=0,
                    )
                    next_to_step3_btn = gr.Button(
                        "Далее",
                        variant="primary",
                        scale=0,
                    )
            
            # ШАГ 3: Запуск анализа
            with gr.Group(visible=False, elem_classes=["step-panel", "disabled"]) as step3_panel:
                gr.Markdown(
                    """
                    <h2 style="margin-bottom: 12px;">Шаг 3: Запуск анализа</h2>
                    <p style="color: #718096; font-size: 14px; margin-bottom: 24px; line-height: 1.6;">
                        После загрузки видео и указания параметров запустите анализ движений. Процесс может занять несколько минут.
                    </p>
                    """
                )
                
                # Индикатор прогресса
                analysis_progress = gr.Progress()
                
                with gr.Row():
                    analyze_btn = gr.Button(
                        "Запустить анализ движений",
                        variant="primary",
                        size="lg",
                        scale=1,
                        interactive=False,
                    )
                    cancel_analysis_btn = gr.Button(
                        "Отменить анализ",
                        variant="stop",
                        size="lg",
                        scale=0,
                        visible=False,
                        interactive=True,
                    )
                
                analysis_status = gr.Markdown(
                    value="<div class='empty-state'><div class='empty-state-title'>Ожидание запуска анализа</div><p>Загрузите видео и укажите параметры для начала анализа</p></div>",
                    visible=True
                )
                
                # Кнопка возврата
                back_to_step2_btn = gr.Button(
                    "Назад",
                    variant="secondary",
                )
            
            # ШАГ 4: Результаты анализа
            with gr.Group(visible=False, elem_classes=["step-panel", "disabled"]) as step4_panel:
                gr.Markdown(
                    """
                    <h2 style="margin-bottom: 12px;">Шаг 4: Результаты анализа</h2>
                    <p style="color: #718096; font-size: 14px; margin-bottom: 24px; line-height: 1.6;">
                        Результаты анализа включают оценку риска, видео с наложенным скелетом и графики реконструкционной ошибки.
                    </p>
                    """
                )
                
                # Текстовый отчет (первым)
                report_output = gr.Textbox(
                    label="Медицинский отчет",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                    container=True,
                    value="Результаты анализа появятся здесь после завершения обработки видео."
                )
                
                gr.Markdown("<hr style='margin: 32px 0;'>")
                
                # Визуальные результаты
                with gr.Row():
                    with gr.Column(scale=1):
                        skeleton_video = gr.Video(
                            label="Видео с наложенным скелетом",
                            height=400,
                            show_label=True,
                        )
                    with gr.Column(scale=1):
                        anomaly_plot = gr.Image(
                            label="График ошибки реконструкции",
                            height=400,
                            show_label=True,
                        )
                
                # Кнопка нового анализа
                new_analysis_btn = gr.Button(
                    "Начать новый анализ",
                    variant="primary",
                    size="lg",
                )
            
            # Статус системы (внизу)
            model_status = gr.Markdown(
                value="<div class='status-info'>Инициализация системы... Загрузка моделей при старте.</div>",
                visible=True,
            )
            
            # Скрытые элементы для управления
            current_user_info = gr.State(value=None)
            
            gr.Markdown("</div>")  # Закрываем контейнер
        
        # Вспомогательная функция для обновления header
        def update_header(user_info_text: str, show_header: bool):
            """Обновить header с информацией о пользователе (БЕЗ кнопки входа, ТОЛЬКО email и кнопка выхода)."""
            if show_header and user_info_text and user_info_text != "Не авторизован" and "Пользователь:" in user_info_text:
                # Извлекаем email из user_info_text
                email = user_info_text.replace("Пользователь: ", "").split(" (")[0]
                return f"""
                <div class="header-panel">
                    <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h1 style="margin: 0; font-size: 20px; font-weight: 600; color: #1a202c;">General Movements Assessment</h1>
                            <p style="margin: 4px 0 0 0; font-size: 13px; color: #718096;">Система анализа движений младенцев</p>
                        </div>
                        <div style="display: flex; align-items: center; gap: 16px;">
                            <span style="font-size: 13px; color: #4a5568;">{email}</span>
                            <button id="header-logout-trigger" onclick="document.getElementById('header-logout-btn').click();" style="background: #e53e3e; border: none; border-radius: 4px; padding: 6px 16px; font-weight: 500; font-size: 13px; color: white; cursor: pointer; transition: background 0.2s ease;">Выйти</button>
                        </div>
                    </div>
                </div>
                """
            return ""
        
        # Вспомогательная функция для определения видимости страниц
        def get_page_visibility(is_auth: bool) -> Tuple[gr.update, gr.update, gr.update]:
            """
            Определить видимость страниц на основе статуса авторизации.
            
            Args:
                is_auth: Авторизован ли пользователь
            
            Returns:
                Tuple (login_page, register_page, main_page)
            """
            logger.info(f"get_page_visibility вызвана с is_auth={is_auth}")
            if is_auth:
                result = (
                    gr.update(visible=False),  # Скрыть страницу входа
                    gr.update(visible=False),  # Скрыть страницу регистрации
                    gr.update(visible=True),   # Показать главную страницу
                )
                logger.info("Возвращаем: login=False, register=False, main=True")
                return result
            else:
                result = (
                    gr.update(visible=True),   # Показать страницу входа
                    gr.update(visible=False),  # Скрыть страницу регистрации
                    gr.update(visible=False),  # Скрыть главную страницу
                )
                logger.info("Возвращаем: login=True, register=False, main=False")
                return result
        
        # Функции аутентификации с использованием AuthHandler и StateManager
        def handle_login(email: str, password: str, current_token, current_auth, current_user) -> Tuple[str, str, bool, Optional[str], bool, Optional[Dict], gr.update, gr.update, gr.update, str, gr.update, str]:
            """Обработка входа пользователя с использованием AuthHandler."""
            if not email or not password:
                return (
                    "<div class='status-error'>Заполните все поля</div>",
                    "Текущий пользователь: Не авторизован",
                    False,
                    current_token,
                    False,
                    None,
                    gr.update(visible=True),  # login_page
                    gr.update(visible=False),  # register_page
                    gr.update(visible=False),  # main_page
                    "<div class='status-info'>Ожидание авторизации...</div>",
                    gr.update(visible=False, value=""),
                    "",
                )
            
            # Используем AuthHandler для входа
            success, message, user_data, session_token = _auth_handler.login(email, password)
            
            if success and user_data and session_token:
                # Обновляем состояние через StateManager
                _state_manager.update_user(
                    is_authenticated=True,
                    session_token=session_token,
                    email=user_data.get('email'),
                    username=user_data.get('username'),
                    full_name=user_data.get('full_name'),
                    role=user_data.get('role', 'user'),
                )
                
                # Переходим на шаг загрузки видео
                _step_manager.go_to_step(AnalysisStep.UPLOAD)
                
                user_info = f"Пользователь: {user_data.get('email', user_data.get('username', ''))}"
                if user_data.get('full_name'):
                    user_info += f" ({user_data['full_name']})"
                
                # Lazy loading моделей (не блокируем вход)
                model_status_text = load_models_and_update_status()
                
                header_html = update_header(user_info, True)
                return (
                    f"<div class='status-success'>{message}</div>",
                    user_info,
                    True,
                    session_token,
                    True,
                    user_data,
                    gr.update(visible=False),  # Скрыть страницу входа
                    gr.update(visible=False),  # Скрыть страницу регистрации
                    gr.update(visible=True),    # Показать главную страницу
                    model_status_text,
                    gr.update(visible=True, value=header_html),    # Показать header
                    header_html,  # Обновить header info
                )
            else:
                # Явно указываем видимость страниц
                return (
                    f"<div class='status-error'>{message}</div>",
                    "Текущий пользователь: Не авторизован",
                    False,
                    current_token,
                    False,
                    None,
                    gr.update(visible=True, value=None),   # login_page - показать
                    gr.update(visible=False, value=None),  # register_page - скрыть
                    gr.update(visible=False, value=None),  # main_page - скрыть
                    "<div class='status-info'>Ожидание авторизации...</div>",
                    gr.update(visible=False, value=""),  # Скрыть header
                    "",  # Header info
                )
        
        def handle_register(
            email: str,
            full_name: str,
            password: str,
            password_confirm: str,
            current_token,
            current_auth,
            current_user
        ) -> Tuple[str, str, bool, Optional[str], bool, Optional[Dict], gr.update, gr.update, gr.update, str, gr.update, str]:
            """Обработка регистрации пользователя с использованием AuthHandler."""
            # Используем AuthHandler для регистрации
            success, message, user_data, session_token = _auth_handler.register(
                email, password, password_confirm, full_name if full_name else None
            )
            
            if not success:
                return (
                    f"<div class='status-error'>{message}</div>",
                    "Текущий пользователь: Не авторизован",
                    False,
                    current_token,
                    False,
                    None,
                    gr.update(visible=False),  # Скрыть страницу входа
                    gr.update(visible=True),    # Показать страницу регистрации
                    gr.update(visible=False),   # Скрыть главную страницу
                    "<div class='status-info'>Ожидание авторизации...</div>",
                    gr.update(visible=False, value=""),
                    "",
                )
            
            if success and session_token and user_data:
                # Обновляем состояние через StateManager
                _state_manager.update_user(
                    is_authenticated=True,
                    session_token=session_token,
                    email=user_data.get('email'),
                    username=user_data.get('username'),
                    full_name=user_data.get('full_name'),
                    role=user_data.get('role', 'user'),
                )
                
                # Переходим на шаг загрузки видео
                _step_manager.go_to_step(AnalysisStep.UPLOAD)
                
                user_info = f"Пользователь: {user_data.get('email', user_data.get('username', ''))}"
                if user_data.get('full_name'):
                    user_info += f" ({user_data['full_name']})"
                
                # Lazy loading моделей
                model_status_text = load_models_and_update_status()
                
                header_html = update_header(user_info, True)
                # Явно указываем видимость страниц
                return (
                    f"<div class='status-success'>{message}</div>",
                    user_info,
                    True,
                    session_token,
                    True,
                    user_data,
                    login_vis,  # Скрыть страницу входа
                    reg_vis,    # Скрыть страницу регистрации
                    main_vis,    # Показать главную страницу
                    model_status_text,
                    gr.update(visible=True, value=header_html),    # Показать header
                    header_html,  # Обновить header info
                )
        
        def toggle_login_register(show_register: bool) -> Tuple[gr.update, gr.update]:
            """Переключение между страницами входа и регистрации."""
            if show_register:
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)
        
        def handle_logout(current_token, current_auth, current_user) -> Tuple[str, bool, Optional[str], bool, Optional[Dict], gr.update, gr.update, gr.update, str, gr.update, str]:
            """Обработка выхода пользователя."""
            if current_token:
                _auth_manager.logout(current_token)
            # Сбрасываем состояние
            _state_manager.update_user(is_authenticated=False, session_token=None)
            _step_manager.go_to_step(AnalysisStep.LOGIN)
            
            # Явно указываем видимость страниц
            return (
                "Текущий пользователь: Не авторизован",
                False,
                None,
                False,
                None,
                gr.update(visible=True, value=None),   # login_page - показать
                gr.update(visible=False, value=None),  # register_page - скрыть
                gr.update(visible=False, value=None),  # main_page - скрыть
                "<div class='status-info'>Ожидание авторизации...</div>",
                gr.update(visible=False, value=""),  # Скрыть header
                "",  # Header info
            )
        
        def check_auth_status(current_token) -> Tuple[str, bool, Optional[str], bool, Optional[Dict], gr.update, gr.update, gr.update, str, gr.update, str]:
            """Проверка статуса авторизации при загрузке страницы с использованием AuthHandler."""
            logger.info(f"check_auth_status вызвана с токеном: {current_token is not None}")
            
            # Сначала проверяем состояние через StateManager
            state = _state_manager.get_state()
            logger.info(f"Состояние из StateManager: is_authenticated={state.user.is_authenticated}, session_token={state.user.session_token is not None}")
            
            # Если пользователь уже авторизован в StateManager, используем его токен
            if state.user.is_authenticated and state.user.session_token:
                current_token = state.user.session_token
                logger.info("Используем токен из StateManager")
                # Если пользователь авторизован в StateManager, сразу возвращаем главную страницу
                user_data = {
                    'email': state.user.email,
                    'username': state.user.username,
                    'full_name': state.user.full_name,
                    'role': state.user.role,
                }
                if user_data.get('email'):
                    logger.info("Пользователь авторизован через StateManager, показываем главную страницу")
                    user_info = f"Пользователь: {user_data.get('email', user_data.get('username', ''))}"
                    if user_data.get('full_name'):
                        user_info += f" ({user_data['full_name']})"
                    
                    model_status_text = load_models_and_update_status()
                    header_html = update_header(user_info, True)
                    logger.info("Возвращаем видимость страниц: login=False, register=False, main=True")
                    # Явно указываем все параметры для gr.update() для надежности
                    return (
                        user_info,
                        True,
                        current_token,
                        True,
                        user_data,
                        gr.update(visible=False, value=None),  # login_page - скрыть
                        gr.update(visible=False, value=None),  # register_page - скрыть
                        gr.update(visible=True, value=None),   # main_page - показать
                        model_status_text,
                        gr.update(visible=True, value=header_html),
                        header_html,
                    )
            
            user_data = _auth_handler.get_user_from_session(current_token)
            logger.info(f"Данные пользователя получены: {user_data is not None}")
            
            if user_data:
                # Обновляем состояние через StateManager
                _state_manager.update_user(
                    is_authenticated=True,
                    session_token=current_token,
                    email=user_data.get('email'),
                    username=user_data.get('username'),
                    full_name=user_data.get('full_name'),
                    role=user_data.get('role', 'user'),
                )
                
                user_info = f"Пользователь: {user_data.get('email', user_data.get('username', ''))}"
                if user_data.get('full_name'):
                    user_info += f" ({user_data['full_name']})"
                
                # Lazy loading моделей
                model_status_text = load_models_and_update_status()
                
                header_html = update_header(user_info, True)
                logger.info(f"Возвращаем видимость страниц: login=False, register=False, main=True")
                # Явно указываем все параметры для gr.update() для надежности
                return (
                    user_info,
                    True,
                    current_token,
                    True,
                    user_data,
                    gr.update(visible=False, value=None),  # login_page - скрыть
                    gr.update(visible=False, value=None),  # register_page - скрыть
                    gr.update(visible=True, value=None),   # main_page - показать
                    model_status_text,
                    gr.update(visible=True, value=header_html),    # Показать header
                    header_html,  # Header info
                )
            else:
                # Сбрасываем состояние
                _state_manager.update_user(is_authenticated=False, session_token=None)
                _step_manager.go_to_step(AnalysisStep.LOGIN)
                
                logger.info(f"Пользователь не авторизован. Видимость страниц: login=True, register=False, main=False")
                # Явно указываем все параметры для gr.update() для надежности
                return (
                    "Текущий пользователь: Не авторизован",
                    False,
                    None,
                    False,
                    None,
                    gr.update(visible=True, value=None),   # login_page - показать
                    gr.update(visible=False, value=None),  # register_page - скрыть
                    gr.update(visible=False, value=None),  # main_page - скрыть
                    "<div class='status-info'>Ожидание авторизации...</div>",
                    gr.update(visible=False, value=""),  # Скрыть header
                    "",  # Header info
                )
        
        # Функции управления шагами
        def update_step_on_video_upload(video_file, current_step_state):
            """Обновить шаг при загрузке видео."""
            if video_file is not None:
                # Видео загружено - активируем шаг 2
                return (
                    2,  # current_step
                    True,  # video_uploaded
                    gr.update(visible=False),  # step1_panel - скрыть
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step2_panel - показать
                    gr.update(visible=False),  # step3_panel - скрыть
                    gr.update(visible=False),  # step4_panel - скрыть
                    "<div class='status-success'>Видео загружено успешно. Перейдите к шагу 2.</div>",  # video_status
                    gr.update(interactive=True)  # analyze_btn
                )
            else:
                return (
                    1,  # current_step
                    False,  # video_uploaded
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step1_panel - показать
                    gr.update(visible=False),  # step2_panel - скрыть
                    gr.update(visible=False),  # step3_panel - скрыть
                    gr.update(visible=False),  # step4_panel - скрыть
                    "<div class='empty-state'><div class='empty-state-title'>Видео не загружено</div><p>Перетащите файл в область выше или нажмите для выбора</p></div>",  # video_status
                    gr.update(interactive=False)  # analyze_btn
                )
        
        def update_step_on_analysis_start():
            """Обновить шаг при запуске анализа с использованием StepManager."""
            global _cancel_event
            # Создаем событие отмены
            _cancel_event = threading.Event()
            _cancel_event.clear()
            
            # Переходим к шагу анализа через StepManager
            success, error = _step_manager.go_to_step(AnalysisStep.ANALYSIS)
            if not success:
                logger.warning(f"Не удалось перейти к шагу анализа: {error}")
            
            # Оптимизация памяти перед анализом
            optimize_memory()
            
            return (
                3,  # current_step для отображения
                gr.update(visible=False),  # step1_panel - скрыть
                gr.update(visible=False),  # step2_panel - скрыть
                gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step3_panel - показать
                gr.update(visible=False),  # step4_panel - скрыть
                "<div class='status-info'>Анализ выполняется. Пожалуйста, подождите...</div>",  # analysis_status
                gr.update(visible=True),  # cancel_analysis_btn - показать
            )
        
        def cancel_analysis():
            """Отменить выполняющийся анализ."""
            global _cancel_event
            if _cancel_event:
                _cancel_event.set()
                logger.info("Запрошена отмена анализа")
                return (
                    "<div class='status-warning'>Анализ отменяется. Пожалуйста, подождите...</div>",
                    gr.update(visible=False),  # cancel_analysis_btn - скрыть
                )
            return (
                "<div class='status-info'>Анализ не выполняется</div>",
                gr.update(visible=False),
            )
        
        def update_step_on_analysis_complete(plot, video, report):
            """Обновить шаг при завершении анализа с использованием StepManager и оптимизацией памяти."""
            global _cancel_event
            # Скрываем кнопку отмены
            cancel_btn_update = gr.update(visible=False)
            
            if plot and video and report:
                # Успешное завершение - переходим к шагу результатов
                success, error = _step_manager.go_to_step(AnalysisStep.RESULTS)
                if not success:
                    logger.warning(f"Не удалось перейти к шагу результатов: {error}")
                
                # Оптимизация памяти после анализа
                optimize_memory()
                
                # Обновляем состояние анализа
                _state_manager.update_analysis(
                    is_running=False,
                    progress=1.0,
                    current_step="Анализ завершен"
                )
                
                return (
                    4,  # current_step
                    gr.update(visible=False),  # step1_panel - скрыть
                    gr.update(visible=False),  # step2_panel - скрыть
                    gr.update(visible=False),  # step3_panel - скрыть
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step4_panel - показать
                    "<div class='status-success'>Анализ завершен. Результаты доступны ниже.</div>",  # analysis_status
                    cancel_btn_update,  # cancel_analysis_btn
                )
            else:
                # Ошибка или отмена
                error_msg = "<div class='status-error'>Анализ завершен с ошибкой или отменен.</div>"
                if _cancel_event and _cancel_event.is_set():
                    error_msg = "<div class='status-warning'>Анализ отменен пользователем.</div>"
                    # При отмене остаемся на шаге анализа
                    _step_manager.go_to_step(AnalysisStep.ANALYSIS)
                
                # Обновляем состояние анализа с ошибкой
                _state_manager.update_analysis(
                    is_running=False,
                    is_cancelled=_cancel_event.is_set() if _cancel_event else False,
                    error=error_msg
                )
                
                # Очистка памяти при ошибке
                optimize_memory()
                
                return (
                    3,  # current_step - остаемся на шаге 3
                    gr.update(visible=False),  # step1_panel
                    gr.update(visible=False),  # step2_panel
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step3_panel
                    gr.update(visible=False),  # step4_panel
                    error_msg,  # analysis_status
                    cancel_btn_update,  # cancel_analysis_btn
                )
        
        # Обработчики событий аутентификации
        # Автоматическая загрузка моделей при старте (только для авторизованных)
        def load_models_and_update_status():
            """Lazy loading моделей с оптимизацией памяти."""
            global _model
            
            # Проверяем, загружены ли уже модели
            if _model is not None:
                state = _state_manager.get_state()
                if state.models.is_loaded:
                    return f"<div class='status-success'><strong>Система готова</strong><br>{state.models.status_message}</div>"
            
            # Очистка памяти перед загрузкой
            optimize_memory()
            
            # Lazy loading моделей (загружаем только при необходимости)
            status = load_models_lazy()
            
            # Обновляем состояние моделей
            state = _state_manager.get_state()
            
            # Форматируем статус для Markdown
            if "успешно" in status.lower() or "готов" in status.lower():
                status_html = f"<div class='status-success'><strong>Система готова</strong><br>{status}</div>"
            elif "ошибка" in status.lower() or "недоступен" in status.lower():
                status_html = f"<div class='status-error'><strong>Ошибка загрузки</strong><br>{status}</div>"
            else:
                status_html = f"<div class='status-info'><strong>{status}</strong></div>"
            
            return status_html
        
        # Периодическая очистка сессий (каждые 24 часа)
        def periodic_cleanup():
            """Периодическая очистка истекших сессий."""
            import time
            while True:
                try:
                    time.sleep(86400)  # 24 часа
                    _auth_manager.cleanup()
                    logger.info("Периодическая очистка сессий выполнена")
                except Exception as e:
                    logger.error(f"Ошибка при очистке сессий: {e}")
        
        # Запускаем периодическую очистку в фоновом потоке
        import threading
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()
        
        # Обработчики событий аутентификации
        login_btn.click(
            fn=handle_login,
            inputs=[login_email, login_password, session_token_storage, is_authenticated, current_user_data],
            outputs=[
                login_status,
                current_user_info,
                is_authenticated,
                session_token_storage,
                is_authenticated,
                current_user_data,
                login_page,
                register_page,
                main_page,
                model_status,
                header_info,
                header_info,  # Для обновления текста в header
            ],
        )
        
        register_btn.click(
            fn=handle_register,
            inputs=[reg_email, reg_full_name, reg_password, reg_password_confirm, session_token_storage, is_authenticated, current_user_data],
            outputs=[
                reg_status,
                current_user_info,
                is_authenticated,
                session_token_storage,
                is_authenticated,
                current_user_data,
                login_page,
                register_page,
                main_page,
                model_status,
                header_info,
                header_info,  # Для обновления текста в header
            ],
        )
        
        logout_btn.click(
            fn=handle_logout,
            inputs=[session_token_storage, is_authenticated, current_user_data],
            outputs=[
                current_user_info,
                is_authenticated,
                session_token_storage,
                is_authenticated,
                current_user_data,
                login_page,
                register_page,
                main_page,
                model_status,
                header_info,
                header_info,  # Для обновления текста в header
            ],
        )
        
        # Функции переключения между страницами
        def show_register():
            return gr.update(visible=False), gr.update(visible=True)
        
        def show_login():
            return gr.update(visible=True), gr.update(visible=False)
        
        # Переключение между страницами входа и регистрации
        show_register_btn.click(
            fn=show_register,
            outputs=[login_page, register_page]
        )
        
        show_login_btn.click(
            fn=show_login,
            outputs=[login_page, register_page]
        )
        
        # Проверка статуса при загрузке интерфейса
        # Используем show_progress=False и queue=False для немедленного выполнения
        # Также добавляем api_name для более надежной работы
        interface.load(
            fn=check_auth_status,
            inputs=[session_token_storage],
            outputs=[
                current_user_info,
                is_authenticated,
                session_token_storage,
                is_authenticated,
                current_user_data,
                login_page,
                register_page,
                main_page,
                model_status,
                header_info,
                header_info,  # Для обновления текста в header
            ],
            show_progress=False,
            queue=False,  # Выполнять немедленно, без очереди
            api_name="check_auth",  # Имя API для отладки
        )
        
        # Дополнительная проверка через 1 секунду после загрузки (на случай, если Gradio перерисовывает компоненты)
        def delayed_auth_check():
            """Дополнительная проверка авторизации с задержкой."""
            import time
            time.sleep(1.0)  # Ждем 1 секунду
            state = _state_manager.get_state()
            if state.user.is_authenticated:
                logger.info("Дополнительная проверка через 1 сек: пользователь авторизован")
                # Обновляем через JavaScript (более надежно)
                return True
            return False
        
        # Запускаем дополнительную проверку в фоне
        import threading
        def run_delayed_check():
            try:
                if delayed_auth_check():
                    logger.info("Дополнительная проверка завершена - пользователь авторизован")
            except Exception as e:
                logger.error(f"Ошибка в дополнительной проверке: {e}")
        
        threading.Thread(target=run_delayed_check, daemon=True).start()
        
        # Дополнительная проверка через 500мс после загрузки (на случай, если Gradio перерисовывает компоненты)
        def delayed_auth_check():
            """Дополнительная проверка авторизации с задержкой."""
            import time
            time.sleep(0.5)
            state = _state_manager.get_state()
            if state.user.is_authenticated:
                logger.info("Дополнительная проверка: пользователь авторизован, обновляем страницы")
                return check_auth_status(state.user.session_token)
            return None
        
        # Запускаем дополнительную проверку в фоне
        import threading
        def run_delayed_check():
            try:
                result = delayed_auth_check()
                if result:
                    # Обновляем компоненты через JavaScript (более надежно)
                    logger.info("Дополнительная проверка завершена")
            except Exception as e:
                logger.error(f"Ошибка в дополнительной проверке: {e}")
        
        threading.Thread(target=run_delayed_check, daemon=True).start()
        
        # Функции навигации между шагами
        def go_to_step(step_num):
            """Переход к указанному шагу с использованием StepManager."""
            # Маппинг числовых шагов на AnalysisStep
            step_mapping = {
                1: AnalysisStep.UPLOAD,
                2: AnalysisStep.PARAMETERS,
                3: AnalysisStep.ANALYSIS,
                4: AnalysisStep.RESULTS,
            }
            
            target_step = step_mapping.get(step_num)
            if target_step:
                # Используем StepManager для перехода
                success, error = _step_manager.go_to_step(target_step)
                if not success:
                    logger.warning(f"Не удалось перейти к шагу {step_num}: {error}")
            
            # Обновляем UI в зависимости от текущего шага
            state = _state_manager.get_state()
            current_analysis_step = state.current_step
            
            updates = {
                AnalysisStep.UPLOAD: (
                    1,
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step1
                    gr.update(visible=False),  # step2
                    gr.update(visible=False),  # step3
                    gr.update(visible=False),  # step4
                ),
                AnalysisStep.PARAMETERS: (
                    2,
                    gr.update(visible=False),  # step1
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step2
                    gr.update(visible=False),  # step3
                    gr.update(visible=False),  # step4
                ),
                AnalysisStep.ANALYSIS: (
                    3,
                    gr.update(visible=False),  # step1
                    gr.update(visible=False),  # step2
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step3
                    gr.update(visible=False),  # step4
                ),
                AnalysisStep.RESULTS: (
                    4,
                    gr.update(visible=False),  # step1
                    gr.update(visible=False),  # step2
                    gr.update(visible=False),  # step3
                    gr.update(visible=True, elem_classes=["step-panel", "active"]),  # step4
                ),
            }
            
            if current_analysis_step in updates:
                step_display, step1_upd, step2_upd, step3_upd, step4_upd = updates[current_analysis_step]
                return (step_display, step1_upd, step2_upd, step3_upd, step4_upd)
            
            # Fallback на шаг 1
            return (
                1,
                gr.update(visible=True, elem_classes=["step-panel", "active"]),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        
        # Обработчики событий для управления шагами
        video_input.change(
            fn=lambda v: gr.update(interactive=bool(v)),
            inputs=[video_input],
            outputs=[next_to_step2_btn]
        )
        
        next_to_step2_btn.click(
            fn=lambda: go_to_step(2),
            outputs=[current_step, step1_panel, step2_panel, step3_panel, step4_panel]
        )
        
        back_to_step1_btn.click(
            fn=lambda: go_to_step(1),
            outputs=[current_step, step1_panel, step2_panel, step3_panel, step4_panel]
        )
        
        next_to_step3_btn.click(
            fn=lambda: go_to_step(3),
            outputs=[current_step, step1_panel, step2_panel, step3_panel, step4_panel]
        )
        
        back_to_step2_btn.click(
            fn=lambda: go_to_step(2),
            outputs=[current_step, step1_panel, step2_panel, step3_panel, step4_panel]
        )
        
        new_analysis_btn.click(
            fn=lambda: go_to_step(1),
            outputs=[current_step, step1_panel, step2_panel, step3_panel, step4_panel]
        )
        
        video_input.change(
            fn=update_step_on_video_upload,
            inputs=[video_input, current_step],
            outputs=[current_step, video_uploaded, step1_panel, step2_panel, step3_panel, step4_panel, video_status, analyze_btn]
        )
        
        # Обработчик запуска анализа
        analyze_btn.click(
            fn=update_step_on_analysis_start,
            inputs=[],
            outputs=[current_step, step1_panel, step2_panel, step3_panel, step4_panel, analysis_status, cancel_analysis_btn]
        ).then(
            fn=analyze_baby_video,
            inputs=[video_input, patient_age_weeks, gestational_age, session_token_storage],
            outputs=[anomaly_plot, skeleton_video, report_output]
        ).then(
            fn=update_step_on_analysis_complete,
            inputs=[anomaly_plot, skeleton_video, report_output],
            outputs=[current_step, step1_panel, step2_panel, step3_panel, step4_panel, analysis_status, cancel_analysis_btn]
        )
        
        # Обработчик отмены анализа
        cancel_analysis_btn.click(
            fn=cancel_analysis,
            inputs=[],
            outputs=[analysis_status, cancel_analysis_btn]
        )
    
    return interface, custom_css


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
    
    interface, custom_css = create_medical_interface()
    
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
            css=custom_css,
            theme=gr.themes.Soft(
                primary_hue="purple",
                secondary_hue="pink",
                neutral_hue="gray",
                font=("ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"),
            )
        )
    except KeyboardInterrupt:
        logger.info("Сервер остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        raise

