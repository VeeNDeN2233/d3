"""
Контроллер анализа видео с управлением шагами и прогрессом.
"""

from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path
import logging
import threading
from enum import Enum

from core.state_manager import AppState, AnalysisStep, AnalysisState
from core.file_processor import VideoProcessor

logger = logging.getLogger(__name__)


class StepManager:
    """Управление шагами интерфейса."""
    
    # Порядок шагов
    STEP_ORDER = [
        AnalysisStep.LOGIN,
        AnalysisStep.UPLOAD,
        AnalysisStep.PARAMETERS,
        AnalysisStep.ANALYSIS,
        AnalysisStep.RESULTS,
    ]
    
    def __init__(self, state_manager: StateManager):
        """
        Инициализация менеджера шагов.
        
        Args:
            state_manager: Менеджер состояний
        """
        self.state_manager = state_manager
    
    def get_state(self) -> AppState:
        """Получить текущее состояние."""
        return self.state_manager.get_state()
    
    def get_current_step(self) -> AnalysisStep:
        """Получить текущий шаг."""
        return self.state_manager.get_state().current_step
    
    def can_go_to_step(self, target_step: AnalysisStep) -> Tuple[bool, Optional[str]]:
        """
        Проверить, можно ли перейти к указанному шагу.
        
        Args:
            target_step: Целевой шаг
        
        Returns:
            Tuple (можно ли перейти, сообщение об ошибке)
        """
        state = self.state_manager.get_state()
        current_index = self.STEP_ORDER.index(state.current_step)
        target_index = self.STEP_ORDER.index(target_step)
        
        # Можно перейти только вперед на один шаг или назад
        if target_index > current_index + 1:
            return False, "Пропуск шагов не разрешен"
        
        # Проверки для конкретных шагов
        if target_step == AnalysisStep.PARAMETERS:
            if not state.video.is_uploaded:
                return False, "Сначала загрузите видео"
        
        if target_step == AnalysisStep.ANALYSIS:
            if not state.video.is_uploaded:
                return False, "Сначала загрузите видео"
            if not state.video.is_valid:
                return False, "Видео не прошло валидацию"
        
        if target_step == AnalysisStep.RESULTS:
            if not state.analysis.results:
                return False, "Анализ еще не завершен"
        
        return True, None
    
    def go_to_step(self, target_step: AnalysisStep) -> Tuple[bool, Optional[str]]:
        """
        Перейти к указанному шагу.
        
        Args:
            target_step: Целевой шаг
        
        Returns:
            Tuple (успех, сообщение об ошибке)
        """
        can_go, error = self.can_go_to_step(target_step)
        if not can_go:
            return False, error
        
        self.state_manager.set_step(target_step)
        return True, None
    
    def next_step(self) -> Tuple[bool, Optional[str]]:
        """
        Перейти к следующему шагу.
        
        Returns:
            Tuple (успех, сообщение об ошибке)
        """
        current_index = self.STEP_ORDER.index(self.state.current_step)
        if current_index >= len(self.STEP_ORDER) - 1:
            return False, "Уже на последнем шаге"
        
        next_step = self.STEP_ORDER[current_index + 1]
        return self.go_to_step(next_step)
    
    def previous_step(self) -> Tuple[bool, Optional[str]]:
        """
        Перейти к предыдущему шагу.
        
        Returns:
            Tuple (успех, сообщение об ошибке)
        """
        current_index = self.STEP_ORDER.index(self.state.current_step)
        if current_index <= 0:
            return False, "Уже на первом шаге"
        
        prev_step = self.STEP_ORDER[current_index - 1]
        return self.go_to_step(prev_step)
    
    def reset_to_start(self):
        """Сбросить к начальному шагу."""
        state = self.state_manager.get_state()
        if state.user.is_authenticated:
            self.state_manager.set_step(AnalysisStep.UPLOAD)
        else:
            self.state_manager.set_step(AnalysisStep.LOGIN)


class AnalysisPipeline:
    """Пайплайн анализа видео с поддержкой отмены и прогресса."""
    
    def __init__(
        self,
        state_manager: StateManager,
        video_processor: VideoProcessor,
        analysis_func: Callable,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ):
        """
        Инициализация пайплайна анализа.
        
        Args:
            state_manager: Менеджер состояний
            video_processor: Обработчик видео
            analysis_func: Функция анализа видео
            progress_callback: Callback для обновления прогресса
            cancel_event: Событие для отмены операции
        """
        self.state_manager = state_manager
        self.video_processor = video_processor
        self.analysis_func = analysis_func
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event or threading.Event()
        self._analysis_thread: Optional[threading.Thread] = None
    
    def _update_progress(self, progress: float, message: str):
        """Обновить прогресс анализа."""
        self.state_manager.update_analysis(progress=progress, current_step=message)
        if self.progress_callback:
            try:
                self.progress_callback(progress, message)
            except Exception as e:
                logger.error(f"Ошибка в progress_callback: {e}")
    
    def start_analysis(
        self,
        video_path: Path,
        patient_age_weeks: int,
        gestational_age: int
    ) -> bool:
        """
        Запустить анализ видео.
        
        Args:
            video_path: Путь к видеофайлу
            patient_age_weeks: Возраст ребенка в неделях
            gestational_age: Гестационный возраст
        
        Returns:
            Успех запуска
        """
        state = self.state_manager.get_state()
        if state.analysis.is_running:
            logger.warning("Анализ уже выполняется")
            return False
        
        # Сброс состояния анализа
        from core.state_manager import AnalysisState
        self.state_manager.update_analysis(
            is_running=True,
            is_cancelled=False,
            progress=0.0,
            current_step="Инициализация анализа",
            error=None,
            results=None
        )
        self.cancel_event.clear()
        
        # Запуск анализа в отдельном потоке
        self._analysis_thread = threading.Thread(
            target=self._run_analysis,
            args=(video_path, patient_age_weeks, gestational_age),
            daemon=True
        )
        self._analysis_thread.start()
        
        return True
    
    def _run_analysis(
        self,
        video_path: Path,
        patient_age_weeks: int,
        gestational_age: int
    ):
        """Выполнить анализ в отдельном потоке."""
        try:
            self._update_progress(0.1, "Инициализация анализа...")
            
            if self.cancel_event.is_set():
                self.state_manager.update_analysis(is_cancelled=True, is_running=False)
                return
            
            self._update_progress(0.2, "Обработка видео...")
            
            # Вызов функции анализа
            results = self.analysis_func(
                video_path,
                patient_age_weeks,
                gestational_age,
                progress_callback=self._update_progress,
                cancel_event=self.cancel_event
            )
            
            if self.cancel_event.is_set():
                self.state_manager.update_analysis(is_cancelled=True, is_running=False)
                return
            
            self._update_progress(0.9, "Формирование результатов...")
            
            # Сохранение результатов
            self.state_manager.update_analysis(
                results=results,
                is_running=False,
                progress=1.0,
                current_step="Анализ завершен"
            )
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении анализа: {e}", exc_info=True)
            self.state_manager.update_analysis(
                error=str(e),
                is_running=False,
                progress=0.0,
                current_step=f"Ошибка: {str(e)}"
            )
    
    def cancel_analysis(self) -> bool:
        """
        Отменить выполняющийся анализ.
        
        Returns:
            Успех отмены
        """
        state = self.state_manager.get_state()
        if not state.analysis.is_running:
            return False
        
        self.cancel_event.set()
        self.state_manager.update_analysis(is_cancelled=True)
        logger.info("Запрошена отмена анализа")
        return True
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Дождаться завершения анализа.
        
        Args:
            timeout: Максимальное время ожидания в секундах
        
        Returns:
            True если анализ завершен, False если прерван по таймауту
        """
        if self._analysis_thread is None:
            return True
        
        self._analysis_thread.join(timeout=timeout)
        return not self._analysis_thread.is_alive()
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        Получить результаты анализа.
        
        Returns:
            Результаты анализа или None
        """
        return self.state_manager.get_state().analysis.results

