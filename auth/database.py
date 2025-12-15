"""
База данных для хранения пользователей.
"""

import sqlite3
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class UserDatabase:
    """Управление базой данных пользователей."""
    
    def __init__(self, db_path: str = "auth/users.db"):
        """
        Инициализация базы данных.
        
        Args:
            db_path: Путь к файлу базы данных SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Инициализация таблиц базы данных."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Таблица пользователей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Таблица сессий
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Индексы для ускорения поиска
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_token ON sessions(session_token)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON sessions(expires_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_active ON users(is_active)")
        
        # Оптимизация SQLite
        cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging для лучшей производительности
        cursor.execute("PRAGMA synchronous=NORMAL")  # Баланс между производительностью и надежностью
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB кэш
        cursor.execute("PRAGMA temp_store=MEMORY")  # Временные таблицы в памяти
        
        conn.commit()
        conn.close()
        logger.info(f"База данных инициализирована: {self.db_path}")
    
    def _hash_password(self, password: str) -> str:
        """Хеширование пароля с использованием SHA-256 и соли."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Проверка пароля."""
        try:
            salt, stored_hash = password_hash.split(':')
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == stored_hash
        except ValueError:
            return False
    
    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        role: str = "user"
    ) -> Tuple[bool, str]:
        """
        Создание нового пользователя.
        
        Args:
            username: Имя пользователя
            password: Пароль
            email: Email (опционально)
            full_name: Полное имя (опционально)
            role: Роль пользователя (user, admin)
        
        Returns:
            Tuple (успех, сообщение)
        """
        # Валидация
        if not username or len(username) < 3:
            return False, "Имя пользователя должно содержать минимум 3 символа"
        
        if not password or len(password) < 6:
            return False, "Пароль должен содержать минимум 6 символов"
        
        if email and '@' not in email:
            return False, "Некорректный email адрес"
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Проверка существования пользователя
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return False, "Пользователь с таким именем уже существует"
            
            if email:
                cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
                if cursor.fetchone():
                    return False, "Пользователь с таким email уже существует"
            
            # Создание пользователя
            password_hash = self._hash_password(password)
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            """, (username, email, password_hash, full_name, role))
            
            conn.commit()
            logger.info(f"Создан новый пользователь: {username}")
            return True, "Пользователь успешно зарегистрирован"
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка создания пользователя: {e}")
            return False, f"Ошибка базы данных: {str(e)}"
        
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Аутентификация пользователя.
        
        Args:
            username: Имя пользователя
            password: Пароль
        
        Returns:
            Tuple (успех, данные пользователя, сообщение)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, username, email, password_hash, full_name, role, is_active
                FROM users WHERE username = ?
            """, (username,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return False, None, "Неверное имя пользователя или пароль"
            
            user_id, db_username, email, password_hash, full_name, role, is_active = user_data
            
            if not is_active:
                return False, None, "Аккаунт деактивирован"
            
            if not self._verify_password(password, password_hash):
                return False, None, "Неверное имя пользователя или пароль"
            
            # Обновление времени последнего входа
            cursor.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            """, (user_id,))
            conn.commit()
            
            user_info = {
                'id': user_id,
                'username': db_username,
                'email': email,
                'full_name': full_name,
                'role': role
            }
            
            logger.info(f"Пользователь аутентифицирован: {username}")
            return True, user_info, "Успешный вход"
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка аутентификации: {e}")
            return False, None, f"Ошибка базы данных: {str(e)}"
        
        finally:
            conn.close()
    
    def create_session(self, user_id: int, duration_hours: int = 24) -> str:
        """
        Создание сессии для пользователя.
        
        Args:
            user_id: ID пользователя
            duration_hours: Длительность сессии в часах
        
        Returns:
            Токен сессии
        """
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            """, (user_id, session_token, expires_at))
            
            conn.commit()
            logger.info(f"Создана сессия для пользователя {user_id}")
            return session_token
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка создания сессии: {e}")
            raise
        
        finally:
            conn.close()
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """
        Проверка валидности сессии.
        
        Args:
            session_token: Токен сессии
        
        Returns:
            Данные пользователя или None
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT s.user_id, s.expires_at, u.username, u.email, u.full_name, u.role, u.is_active
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
            """, (session_token,))
            
            session_data = cursor.fetchone()
            
            if not session_data:
                return None
            
            user_id, expires_at, username, email, full_name, role, is_active = session_data
            
            if not is_active:
                return None
            
            return {
                'id': user_id,
                'username': username,
                'email': email,
                'full_name': full_name,
                'role': role,
                'expires_at': expires_at
            }
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка проверки сессии: {e}")
            return None
        
        finally:
            conn.close()
    
    def delete_session(self, session_token: str) -> bool:
        """
        Удаление сессии (выход).
        
        Args:
            session_token: Токен сессии
        
        Returns:
            Успех операции
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
            conn.commit()
            logger.info(f"Сессия удалена: {session_token[:10]}...")
            return True
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка удаления сессии: {e}")
            return False
        
        finally:
            conn.close()
    
    def cleanup_expired_sessions(self, batch_size: int = 1000):
        """Очистка истекших сессий батчами для оптимизации."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Используем батчинг для больших объемов данных
            total_deleted = 0
            while True:
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE expires_at < CURRENT_TIMESTAMP 
                    AND id IN (
                        SELECT id FROM sessions 
                        WHERE expires_at < CURRENT_TIMESTAMP 
                        LIMIT ?
                    )
                """, (batch_size,))
                deleted = cursor.rowcount
                total_deleted += deleted
                conn.commit()
                
                if deleted == 0:
                    break
            
            if total_deleted > 0:
                logger.info(f"Удалено {total_deleted} истекших сессий")
                # Оптимизация базы данных после удаления
                cursor.execute("VACUUM")
                conn.commit()
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка очистки сессий: {e}")
        
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Получить данные пользователя по ID."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, username, email, full_name, role, created_at, last_login, is_active
                FROM users WHERE id = ?
            """, (user_id,))
            
            user_data = cursor.fetchone()
            if not user_data:
                return None
            
            return {
                'id': user_data[0],
                'username': user_data[1],
                'email': user_data[2],
                'full_name': user_data[3],
                'role': user_data[4],
                'created_at': user_data[5],
                'last_login': user_data[6],
                'is_active': user_data[7]
            }
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения пользователя: {e}")
            return None
        
        finally:
            conn.close()

