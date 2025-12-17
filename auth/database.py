"""
База данных для хранения пользователей.
С защитой от SQL инъекций через параметризованные запросы.
"""

import sqlite3
import bcrypt
import hashlib  # Для обратной совместимости со старыми паролями
import secrets
from pathlib import Path
from typing import Optional, Dict, Tuple, List
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
        """
        Безопасное хэширование пароля с использованием bcrypt.
        Bcrypt автоматически добавляет соль и использует адаптивный алгоритм.
        """
        # Генерируем соль и хэшируем пароль
        salt = bcrypt.gensalt(rounds=12)  # 12 раундов для баланса безопасности и производительности
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Проверка пароля с поддержкой старых (SHA-256) и новых (bcrypt) форматов.
        Bcrypt автоматически извлекает соль из хэша.
        """
        try:
            # Пытаемся проверить как bcrypt (новый формат)
            if password_hash.startswith('$2b$') or password_hash.startswith('$2a$'):
                return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
            
            # Старый формат SHA-256 (для обратной совместимости)
            # Формат: "salt:hash"
            if ':' in password_hash:
                salt, stored_hash = password_hash.split(':', 1)
                computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
                return computed_hash == stored_hash
            
            # Если формат не распознан, пробуем bcrypt
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
            
        except (ValueError, TypeError) as e:
            logger.error(f"Ошибка проверки пароля: {e}")
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
            username: Имя пользователя или email
            password: Пароль
        
        Returns:
            Tuple (успех, данные пользователя, сообщение)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Пытаемся найти пользователя по username или email
            cursor.execute("""
                SELECT id, username, email, password_hash, full_name, role, is_active
                FROM users WHERE username = ? OR email = ?
            """, (username, username))
            
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
                'is_active': bool(user_data[7])
            }
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения пользователя: {e}")
            return None
        
        finally:
            conn.close()
    
    def get_all_users(self) -> List[Dict]:
        """Получить список всех пользователей (для админки)."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, username, email, full_name, role, created_at, last_login, is_active
                FROM users
                ORDER BY created_at DESC
            """)
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'full_name': row[3],
                    'role': row[4],
                    'created_at': row[5],
                    'last_login': row[6],
                    'is_active': bool(row[7])
                })
            
            return users
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения списка пользователей: {e}")
            return []
        
        finally:
            conn.close()
    
    def update_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
        password: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Обновление данных пользователя (для админки).
        Использует параметризованные запросы для защиты от SQL инъекций.
        
        Args:
            user_id: ID пользователя
            username: Новое имя пользователя
            email: Новый email
            full_name: Новое полное имя
            role: Новая роль
            is_active: Активен ли пользователь
            password: Новый пароль (опционально)
        
        Returns:
            Tuple (успех, сообщение)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Проверяем существование пользователя (параметризованный запрос)
            cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            if not cursor.fetchone():
                return False, "Пользователь не найден"
            
            updates = []
            params = []
            
            if username is not None:
                # Проверяем уникальность username (параметризованный запрос)
                cursor.execute("SELECT id FROM users WHERE username = ? AND id != ?", (username, user_id))
                if cursor.fetchone():
                    return False, "Пользователь с таким именем уже существует"
                updates.append("username = ?")
                params.append(username)
            
            if email is not None:
                # Проверяем уникальность email (параметризованный запрос)
                cursor.execute("SELECT id FROM users WHERE email = ? AND id != ?", (email, user_id))
                if cursor.fetchone():
                    return False, "Пользователь с таким email уже существует"
                updates.append("email = ?")
                params.append(email)
            
            if full_name is not None:
                updates.append("full_name = ?")
                params.append(full_name)
            
            if role is not None:
                if role not in ['user', 'admin']:
                    return False, "Недопустимая роль"
                updates.append("role = ?")
                params.append(role)
            
            if is_active is not None:
                updates.append("is_active = ?")
                params.append(1 if is_active else 0)
            
            if password is not None:
                if len(password) < 6:
                    return False, "Пароль должен содержать минимум 6 символов"
                password_hash = self._hash_password(password)
                updates.append("password_hash = ?")
                params.append(password_hash)
            
            if not updates:
                return False, "Нет данных для обновления"
            
            params.append(user_id)
            # Параметризованный запрос - защита от SQL инъекций
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
            
            cursor.execute(query, params)
            conn.commit()
            
            logger.info(f"Пользователь {user_id} обновлен")
            return True, "Пользователь успешно обновлен"
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка обновления пользователя: {e}")
            return False, f"Ошибка базы данных: {str(e)}"
        
        finally:
            conn.close()
    
    def delete_user(self, user_id: int) -> Tuple[bool, str]:
        """
        Удаление пользователя (для админки).
        Использует параметризованные запросы для защиты от SQL инъекций.
        
        Args:
            user_id: ID пользователя
        
        Returns:
            Tuple (успех, сообщение)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Проверяем существование пользователя (параметризованный запрос)
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            user_data = cursor.fetchone()
            if not user_data:
                return False, "Пользователь не найден"
            
            username = user_data[0]
            
            # Удаляем все сессии пользователя (параметризованный запрос)
            cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            
            # Удаляем пользователя (параметризованный запрос)
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            
            logger.info(f"Пользователь {username} (ID: {user_id}) удален")
            return True, f"Пользователь {username} успешно удален"
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка удаления пользователя: {e}")
            return False, f"Ошибка базы данных: {str(e)}"
        
        finally:
            conn.close()

