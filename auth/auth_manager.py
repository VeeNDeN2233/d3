"""
Менеджер аутентификации и авторизации.
"""

import secrets
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import logging

from .database import UserDatabase

logger = logging.getLogger(__name__)


class AuthManager:
    """Управление аутентификацией и сессиями."""
    
    def __init__(self, db_path: str = "auth/users.db"):
        """
        Инициализация менеджера аутентификации.
        
        Args:
            db_path: Путь к базе данных пользователей
        """
        self.db = UserDatabase(db_path)
        # Хранилище активных сессий в памяти (для быстрого доступа)
        self._active_sessions: Dict[str, Dict] = {}
    
    def register(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Регистрация нового пользователя.
        
        Args:
            username: Имя пользователя
            password: Пароль
            email: Email (опционально)
            full_name: Полное имя (опционально)
        
        Returns:
            Tuple (успех, сообщение, токен сессии)
        """
        success, message = self.db.create_user(username, password, email, full_name)
        
        if success:
            # Автоматический вход после регистрации
            auth_success, user_data, auth_message = self.db.authenticate_user(username, password)
            if auth_success:
                session_token = self.db.create_session(user_data['id'])
                self._active_sessions[session_token] = {
                    'user': user_data,
                    'created_at': datetime.now()
                }
                return True, message, session_token
        
        return False, message, None
    
    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[str], Optional[Dict]]:
        """
        Вход пользователя.
        
        Args:
            username: Имя пользователя
            password: Пароль
        
        Returns:
            Tuple (успех, сообщение, токен сессии, данные пользователя)
        """
        auth_success, user_data, message = self.db.authenticate_user(username, password)
        
        if auth_success:
            session_token = self.db.create_session(user_data['id'])
            self._active_sessions[session_token] = {
                'user': user_data,
                'created_at': datetime.now()
            }
            logger.info(f"Пользователь {username} вошел в систему")
            return True, message, session_token, user_data
        
        return False, message, None, None
    
    def logout(self, session_token: str) -> bool:
        """
        Выход пользователя.
        
        Args:
            session_token: Токен сессии
        
        Returns:
            Успех операции
        """
        if session_token in self._active_sessions:
            del self._active_sessions[session_token]
        
        return self.db.delete_session(session_token)
    
    def get_user_from_session(self, session_token: Optional[str]) -> Optional[Dict]:
        """
        Получить данные пользователя из сессии.
        
        Args:
            session_token: Токен сессии
        
        Returns:
            Данные пользователя или None
        """
        if not session_token:
            return None
        
        # Сначала проверяем в памяти
        if session_token in self._active_sessions:
            session_data = self._active_sessions[session_token]
            # Проверяем, не истекла ли сессия (24 часа)
            if (datetime.now() - session_data['created_at']).total_seconds() < 86400:
                return session_data['user']
            else:
                # Сессия истекла, удаляем из памяти
                del self._active_sessions[session_token]
        
        # Проверяем в базе данных
        user_data = self.db.validate_session(session_token)
        if user_data:
            # Обновляем кэш
            self._active_sessions[session_token] = {
                'user': user_data,
                'created_at': datetime.now()
            }
            return user_data
        
        return None
    
    def is_authenticated(self, session_token: Optional[str]) -> bool:
        """
        Проверка, аутентифицирован ли пользователь.
        
        Args:
            session_token: Токен сессии
        
        Returns:
            True если пользователь аутентифицирован
        """
        return self.get_user_from_session(session_token) is not None
    
    def require_auth(self, session_token: Optional[str]) -> Tuple[bool, Optional[Dict], str]:
        """
        Требовать аутентификацию (для защиты функций).
        
        Args:
            session_token: Токен сессии
        
        Returns:
            Tuple (успех, данные пользователя, сообщение об ошибке)
        """
        user_data = self.get_user_from_session(session_token)
        
        if not user_data:
            return False, None, "Требуется авторизация. Пожалуйста, войдите в систему."
        
        return True, user_data, ""
    
    def require_role(self, session_token: Optional[str], required_role: str = "admin") -> Tuple[bool, Optional[Dict], str]:
        """
        Требовать определенную роль.
        
        Args:
            session_token: Токен сессии
            required_role: Требуемая роль
        
        Returns:
            Tuple (успех, данные пользователя, сообщение об ошибке)
        """
        success, user_data, message = self.require_auth(session_token)
        
        if not success:
            return False, None, message
        
        if user_data.get('role') != required_role:
            return False, user_data, f"Требуется роль: {required_role}"
        
        return True, user_data, ""
    
    def cleanup(self):
        """Очистка истекших сессий."""
        self.db.cleanup_expired_sessions()
        
        # Очистка кэша в памяти
        now = datetime.now()
        expired_tokens = [
            token for token, session_data in self._active_sessions.items()
            if (now - session_data['created_at']).total_seconds() >= 86400
        ]
        for token in expired_tokens:
            del self._active_sessions[token]
        
        logger.info(f"Очищено {len(expired_tokens)} истекших сессий из памяти")

