"""
Адаптер для интеграции StateManager с Gradio компонентами.
"""

from typing import Any, Dict, Optional, Tuple
import logging
from core.state_manager import StateManager, AppState, AnalysisStep

logger = logging.getLogger(__name__)


class GradioStateAdapter:
    """Адаптер для синхронизации StateManager с Gradio State компонентами."""
    
    def __init__(self, state_manager: StateManager):
        """
        Инициализация адаптера.
        
        Args:
            state_manager: Менеджер состояний
        """
        self.state_manager = state_manager
    
    def sync_to_gradio_state(self) -> Dict[str, Any]:
        """
        Синхронизировать состояние из StateManager в формат для Gradio State.
        
        Returns:
            Словарь с состояниями для Gradio
        """
        state = self.state_manager.get_state()
        return {
            'session_token': state.user.session_token,
            'is_authenticated': state.user.is_authenticated,
            'current_user_data': {
                'email': state.user.email,
                'username': state.user.username,
                'full_name': state.user.full_name,
                'role': state.user.role,
            } if state.user.is_authenticated else None,
            'current_step': state.current_step.value if isinstance(state.current_step, AnalysisStep) else state.current_step,
            'video_uploaded': state.video.is_uploaded,
        }
    
    def sync_from_gradio_state(self, gradio_state: Dict[str, Any]):
        """
        Синхронизировать состояние из Gradio State в StateManager.
        
        Args:
            gradio_state: Словарь состояний из Gradio
        """
        if 'session_token' in gradio_state:
            self.state_manager.update_user(session_token=gradio_state['session_token'])
        
        if 'is_authenticated' in gradio_state:
            self.state_manager.update_user(is_authenticated=gradio_state['is_authenticated'])
        
        if 'current_user_data' in gradio_state and gradio_state['current_user_data']:
            user_data = gradio_state['current_user_data']
            self.state_manager.update_user(
                email=user_data.get('email'),
                username=user_data.get('username'),
                full_name=user_data.get('full_name'),
                role=user_data.get('role', 'user'),
            )
        
        if 'current_step' in gradio_state:
            try:
                step = AnalysisStep(gradio_state['current_step'])
                self.state_manager.set_step(step)
            except (ValueError, TypeError):
                # Если не удалось распарсить, игнорируем
                pass
        
        if 'video_uploaded' in gradio_state:
            self.state_manager.update_video(is_uploaded=gradio_state['video_uploaded'])
    
    def get_state_for_gradio(self) -> Tuple[Optional[str], bool, Optional[Dict], str, bool]:
        """
        Получить состояние в формате для Gradio компонентов.
        
        Returns:
            Tuple (session_token, is_authenticated, user_data, current_step, video_uploaded)
        """
        state = self.state_manager.get_state()
        return (
            state.user.session_token,
            state.user.is_authenticated,
            {
                'email': state.user.email,
                'username': state.user.username,
                'full_name': state.user.full_name,
                'role': state.user.role,
            } if state.user.is_authenticated else None,
            state.current_step.value if isinstance(state.current_step, AnalysisStep) else str(state.current_step),
            state.video.is_uploaded,
        )

