
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
    def wrapped_analysis(
        video_file: Any,
        age_weeks: Optional[int] = None,
        gestational_age_weeks: Optional[int] = None,
        session_token: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:

            if progress_callback:
                progress_callback(0.1, "Обработка файла...")
            

            if cancel_event and cancel_event.is_set():
                state.analysis.is_cancelled = True
                return None, None, "Анализ отменен пользователем"
            

            video_path = video_processor.get_video_path(video_file)
            
            if video_path is None:
                return None, None, "Ошибка: Не удалось определить путь к видеофайлу"
            

            is_valid, error_msg = video_processor.validate_video(video_path)
            if not is_valid:
                return None, None, f"Ошибка валидации видео: {error_msg}"
            

            state.video.file_path = str(video_path)
            state.video.file_name = video_path.name
            state.video.file_size = video_path.stat().st_size
            state.video.is_uploaded = True
            state.video.is_valid = True
            
            if progress_callback:
                progress_callback(0.2, "Запуск анализа...")
            

            if cancel_event and cancel_event.is_set():
                state.analysis.is_cancelled = True
                return None, None, "Анализ отменен пользователем"
            



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

