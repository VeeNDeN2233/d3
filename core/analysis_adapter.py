"""
Адаптер для интеграции существующей функции анализа с новой архитектурой.
"""

from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path
import logging
import threading

from core.state_manager import AppState
from core.file_processor import VideoProcessor

logger = logging.getLogger(__name__)


def create_analysis_wrapper(
    original_analysis_func: Callable,
    video_processor: VideoProcessor,
    state: AppState,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> Callable:
    """
    Создать обертку для функции анализа с поддержкой прогресса и отмены.
    
    Args:
        original_analysis_func: Оригинальная функция анализа
        video_processor: Обработчик видео
        state: Состояние приложения
        progress_callback: Callback для обновления прогресса
        cancel_event: Событие для отмены операции
    
    Returns:
        Обернутая функция анализа
    """
    def wrapped_analysis(
        video_file: Any,
        age_weeks: Optional[int] = None,
        gestational_age_weeks: Optional[int] = None,
        session_token: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Обернутая функция анализа с обработкой файлов и прогрессом.
        
        Args:
            video_file: Файл от Gradio
            age_weeks: Возраст ребенка в неделях
            gestational_age_weeks: Гестационный возраст
            session_token: Токен сессии
        
        Returns:
            Tuple (plot_path, video_path, report_text)
        """
        try:
            # Обновление прогресса
            if progress_callback:
                progress_callback(0.1, "Обработка файла...")
            
            # Проверка отмены
            if cancel_event and cancel_event.is_set():
                state.analysis.is_cancelled = True
                return None, None, "Анализ отменен пользователем"
            
            # Получение пути к видео через VideoProcessor
            video_path = video_processor.get_video_path(video_file)
            
            if video_path is None:
                return None, None, "Ошибка: Не удалось определить путь к видеофайлу"
            
            # Валидация файла
            is_valid, error_msg = video_processor.validate_video(video_path)
            if not is_valid:
                return None, None, f"Ошибка валидации видео: {error_msg}"
            
            # Обновление состояния
            state.video.file_path = str(video_path)
            state.video.file_name = video_path.name
            state.video.file_size = video_path.stat().st_size
            state.video.is_uploaded = True
            state.video.is_valid = True
            
            if progress_callback:
                progress_callback(0.2, "Запуск анализа...")
            
            # Проверка отмены
            if cancel_event and cancel_event.is_set():
                state.analysis.is_cancelled = True
                return None, None, "Анализ отменен пользователем"
            
            # Вызов оригинальной функции анализа
            # Оригинальная функция ожидает video_file, но мы передаем путь
            # Нужно адаптировать вызов
            result = original_analysis_func(
                str(video_path),
                age_weeks,
                gestational_age_weeks,
                session_token
            )
            
            if progress_callback:
                progress_callback(1.0, "Анализ завершен")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в обернутой функции анализа: {e}", exc_info=True)
            state.analysis.error = str(e)
            return None, None, f"Ошибка анализа: {str(e)}"
    
    return wrapped_analysis

