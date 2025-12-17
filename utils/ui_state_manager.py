"""
Менеджер состояния UI для Gradio интерфейса.
Управляет видимостью и созданием компонентов на основе состояния авторизации.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from core.state_manager import StateManager, AppState

logger = logging.getLogger(__name__)


class UIStateManager:
    """Менеджер состояния UI для условного отображения компонентов."""
    
    def __init__(self, state_manager: StateManager):
        """
        Инициализация менеджера UI состояния.
        
        Args:
            state_manager: Менеджер состояния приложения
        """
        self.state_manager = state_manager
    
    def should_show_login_page(self) -> bool:
        """Определить, нужно ли показывать страницу входа."""
        state = self.state_manager.get_state()
        return not state.user.is_authenticated
    
    def should_show_main_page(self) -> bool:
        """Определить, нужно ли показывать главную страницу."""
        state = self.state_manager.get_state()
        return state.user.is_authenticated
    
    def get_page_visibility(self) -> Tuple[bool, bool, bool]:
        """
        Получить видимость страниц.
        
        Returns:
            Tuple (show_login, show_register, show_main)
        """
        state = self.state_manager.get_state()
        is_auth = state.user.is_authenticated
        
        return (
            not is_auth,  # login_page
            False,        # register_page (всегда скрыта, показывается только при переключении)
            is_auth,      # main_page
        )
