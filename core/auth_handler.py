
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

from auth.auth_manager import AuthManager
from core.state_manager import UserState, AppState

logger = logging.getLogger(__name__)


class AuthHandler:
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = "auth/users.db"
        self.auth_manager = AuthManager(db_path)
    
    def login(self, email: str, password: str) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
        if not email or not password:
            return False, "Заполните все поля", None, None
        
        try:
            success, message, session_token, user_data = self.auth_manager.login(email, password)
            
            if success and user_data:
                return True, message, user_data, session_token
            else:
                return False, message or "Неверное имя пользователя или пароль", None, None
                
        except Exception as e:
            logger.error(f"Ошибка входа: {e}", exc_info=True)
            return False, f"Ошибка входа: {str(e)}", None, None
    
    def register(
        self,
        email: str,
        password: str,
        password_confirm: str,
        full_name: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
        if not email or not password:
            return False, "Заполните email и пароль", None, None
        
        if password != password_confirm:
            return False, "Пароли не совпадают", None, None
        
        if len(password) < 6:
            return False, "Пароль должен содержать минимум 6 символов", None, None
        
        try:

            username = email.split('@')[0]
            
            success, message, session_token = self.auth_manager.register(
                username, password, email, full_name
            )
            
            if success and session_token:

                auth_success, user_data, _ = self.auth_manager.require_auth(session_token)
                if auth_success:
                    return True, message, user_data, session_token
            
            return False, message or "Ошибка регистрации", None, None
            
        except Exception as e:
            logger.error(f"Ошибка регистрации: {e}", exc_info=True)
            return False, f"Ошибка регистрации: {str(e)}", None, None
    
    def logout(self, session_token: Optional[str]) -> bool:
        if not session_token:
            return True
        
        try:
            return self.auth_manager.logout(session_token)
        except Exception as e:
            logger.error(f"Ошибка выхода: {e}", exc_info=True)
            return False
    
    def get_user_from_session(self, session_token: Optional[str]) -> Optional[Dict]:
        if not session_token:
            return None
        
        try:
            return self.auth_manager.get_user_from_session(session_token)
        except Exception as e:
            logger.error(f"Ошибка получения пользователя: {e}", exc_info=True)
            return None
    
    def update_user_state(self, state: AppState, session_token: Optional[str]) -> AppState:
        user_data = self.get_user_from_session(session_token)
        
        if user_data:
            state.user.is_authenticated = True
            state.user.session_token = session_token
            state.user.email = user_data.get('email')
            state.user.username = user_data.get('username')
            state.user.full_name = user_data.get('full_name')
            state.user.role = user_data.get('role', 'user')
        else:
            state.user = UserState()
        
        return state

