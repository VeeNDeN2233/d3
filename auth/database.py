
import sqlite3
import bcrypt
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class UserDatabase:
    
    def __init__(self, db_path: str = "auth/users.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        

        

        

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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                last_name TEXT NOT NULL,
                first_name TEXT NOT NULL,
                middle_name TEXT,
                birth_date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_id INTEGER NOT NULL,
                video_filename TEXT,
                output_dir TEXT NOT NULL,
                age_weeks INTEGER,
                gestational_age INTEGER,
                report_text TEXT,
                is_anomaly INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_token ON sessions(session_token)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON sessions(expires_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_active ON users(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_user_id ON patients(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_user_id ON analysis_results(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_patient_id ON analysis_results(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_created_at ON analysis_results(created_at)")
        

        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        conn.commit()
        conn.close()
        logger.info(f"База данных инициализирована: {self.db_path}")
    
    def _hash_password(self, password: str) -> str:

        salt = bcrypt.gensalt(rounds=12)
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:

            if password_hash.startswith('$2b$') or password_hash.startswith('$2a$'):
                return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
            


            if ':' in password_hash:
                salt, stored_hash = password_hash.split(':', 1)
                computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
                return computed_hash == stored_hash
            

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

        if not username or len(username) < 3:
            return False, "Имя пользователя должно содержать минимум 3 символа"
        
        if not password or len(password) < 6:
            return False, "Пароль должен содержать минимум 6 символов"
        
        if email and '@' not in email:
            return False, "Некорректный email адрес"
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:

            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return False, "Пользователь с таким именем уже существует"
            
            if email:
                cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
                if cursor.fetchone():
                    return False, "Пользователь с таким email уже существует"
            
            password_hash = self._hash_password(password)
            logger.info(f"Создание пользователя: username={username}, email={email}, hash_length={len(password_hash)}")
            
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            """, (username, email, password_hash, full_name, role))
            
            conn.commit()
            user_id = cursor.lastrowid
            logger.info(f"Создан новый пользователь: {username} (ID: {user_id})")
            
            cursor.execute("SELECT id, username, email, password_hash FROM users WHERE id = ?", (user_id,))
            verify_user = cursor.fetchone()
            if verify_user:
                logger.info(f"Проверка: пользователь создан успешно - ID: {verify_user[0]}, username: {verify_user[1]}, email: {verify_user[2]}")
            else:
                logger.error(f"ОШИБКА: пользователь не найден после создания!")
            
            return True, "Пользователь успешно зарегистрирован"
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка создания пользователя: {e}")
            return False, f"Ошибка базы данных: {str(e)}"
        
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict], str]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            logger.info(f"Попытка аутентификации: username/email={username}")
            cursor.execute("""
                SELECT id, username, email, password_hash, full_name, role, is_active
                FROM users WHERE username = ? OR email = ?
            """, (username, username))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                logger.warning(f"Пользователь не найден: {username}")
                return False, None, "Неверное имя пользователя или пароль"
            
            user_id, db_username, email, password_hash, full_name, role, is_active = user_data
            logger.info(f"Найден пользователь: ID={user_id}, username={db_username}, email={email}, is_active={is_active}")
            
            if not is_active:
                logger.warning(f"Аккаунт деактивирован: {username}")
                return False, None, "Аккаунт деактивирован"
            
            password_valid = self._verify_password(password, password_hash)
            logger.info(f"Проверка пароля: valid={password_valid}, hash_prefix={password_hash[:30]}...")
            if not password_valid:
                logger.warning(f"Неверный пароль для пользователя {username}")
                return False, None, "Неверное имя пользователя или пароль"
            

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
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            
            conn.commit()
            logger.info(f"Создана сессия для пользователя {user_id}")
            return session_token
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка создания сессии: {e}")
            raise
        
        finally:
            conn.close()
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            
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
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:

            total_deleted = 0
            while True:
                deleted = cursor.rowcount
                total_deleted += deleted
                conn.commit()
                
                if deleted == 0:
                    break
            
            if total_deleted > 0:
                logger.info(f"Удалено {total_deleted} истекших сессий")

                cursor.execute("VACUUM")
                conn.commit()
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка очистки сессий: {e}")
        
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
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
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, username, email, full_name, role, created_at, last_login, is_active
                FROM users WHERE username = ?
            """, (username,))
            
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
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:

            cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            if not cursor.fetchone():
                return False, "Пользователь не найден"
            
            updates = []
            params = []
            
            if username is not None:

                cursor.execute("SELECT id FROM users WHERE username = ? AND id != ?", (username, user_id))
                if cursor.fetchone():
                    return False, "Пользователь с таким именем уже существует"
                updates.append("username = ?")
                params.append(username)
            
            if email is not None:

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
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:

            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            user_data = cursor.fetchone()
            if not user_data:
                return False, "Пользователь не найден"
            
            username = user_data[0]
            

            cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            

            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            
            logger.info(f"Пользователь {username} (ID: {user_id}) удален")
            return True, f"Пользователь {username} успешно удален"
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка удаления пользователя: {e}")
            return False, f"Ошибка базы данных: {str(e)}"
        
        finally:
            conn.close()
    
    def create_patient(
        self,
        user_id: int,
        last_name: str,
        first_name: str,
        middle_name: Optional[str],
        birth_date: str
    ) -> Tuple[bool, int, str]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO patients (user_id, last_name, first_name, middle_name, birth_date)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, last_name, first_name, middle_name, birth_date))
            conn.commit()
            patient_id = cursor.lastrowid
            logger.info(f"Создан пациент {last_name} {first_name} для пользователя {user_id}")
            return True, patient_id, "Пациент успешно создан"
        except sqlite3.Error as e:
            logger.error(f"Ошибка создания пациента: {e}")
            return False, 0, f"Ошибка базы данных: {str(e)}"
        finally:
            conn.close()
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, user_id, last_name, first_name, middle_name, birth_date, created_at
                FROM patients WHERE id = ?
            """, (patient_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'id': row[0],
                'user_id': row[1],
                'last_name': row[2],
                'first_name': row[3],
                'middle_name': row[4],
                'birth_date': row[5],
                'created_at': row[6]
            }
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения пациента: {e}")
            return None
        finally:
            conn.close()
    
    def create_analysis_result(
        self,
        user_id: int,
        patient_id: int,
        video_filename: Optional[str],
        output_dir: str,
        age_weeks: Optional[int],
        gestational_age: Optional[int],
        report_text: Optional[str],
        is_anomaly: bool = False
    ) -> Tuple[bool, int, str]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO analysis_results 
                (user_id, patient_id, video_filename, output_dir, age_weeks, gestational_age, report_text, is_anomaly)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, patient_id, video_filename, output_dir, age_weeks, gestational_age, report_text, 1 if is_anomaly else 0))
            conn.commit()
            result_id = cursor.lastrowid
            logger.info(f"Создан результат анализа {result_id} для пользователя {user_id}")
            return True, result_id, "Результат анализа успешно сохранен"
        except sqlite3.Error as e:
            logger.error(f"Ошибка создания результата анализа: {e}")
            return False, 0, f"Ошибка базы данных: {str(e)}"
        finally:
            conn.close()
    
    def get_user_analysis_results(self, user_id: int, limit: int = 100) -> List[Dict]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    ar.id, ar.patient_id, ar.video_filename, ar.output_dir, 
                    ar.age_weeks, ar.gestational_age, ar.report_text, ar.is_anomaly, ar.created_at,
                    p.last_name, p.first_name, p.middle_name, p.birth_date
                FROM analysis_results ar
                JOIN patients p ON ar.patient_id = p.id
                WHERE ar.user_id = ?
                ORDER BY ar.created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'patient_id': row[1],
                    'video_filename': row[2],
                    'output_dir': row[3],
                    'age_weeks': row[4],
                    'gestational_age': row[5],
                    'report_text': row[6],
                    'is_anomaly': bool(row[7]),
                    'created_at': row[8],
                    'patient_last_name': row[9],
                    'patient_first_name': row[10],
                    'patient_middle_name': row[11],
                    'patient_birth_date': row[12]
                })
            
            return results
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения результатов анализа: {e}")
            return []
        finally:
            conn.close()
    
    def get_analysis_result_by_id(self, result_id: int, user_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    ar.id, ar.patient_id, ar.video_filename, ar.output_dir, 
                    ar.age_weeks, ar.gestational_age, ar.report_text, ar.is_anomaly, ar.created_at,
                    p.last_name, p.first_name, p.middle_name, p.birth_date
                FROM analysis_results ar
                JOIN patients p ON ar.patient_id = p.id
                WHERE ar.id = ? AND ar.user_id = ?
            """, (result_id, user_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'id': row[0],
                'patient_id': row[1],
                'video_filename': row[2],
                'output_dir': row[3],
                'age_weeks': row[4],
                'gestational_age': row[5],
                'report_text': row[6],
                'is_anomaly': bool(row[7]),
                'created_at': row[8],
                'patient_last_name': row[9],
                'patient_first_name': row[10],
                'patient_middle_name': row[11],
                'patient_birth_date': row[12]
            }
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения результата анализа: {e}")
            return None
        finally:
            conn.close()
    
    def delete_analysis_result(self, result_id: int, user_id: int) -> Tuple[bool, str]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM analysis_results WHERE id = ? AND user_id = ?", (result_id, user_id))
            if not cursor.fetchone():
                return False, "Результат анализа не найден или у вас нет прав на его удаление"
            
            cursor.execute("DELETE FROM analysis_results WHERE id = ? AND user_id = ?", (result_id, user_id))
            conn.commit()
            
            logger.info(f"Результат анализа {result_id} удален пользователем {user_id}")
            return True, "Результат анализа успешно удален"
        except sqlite3.Error as e:
            logger.error(f"Ошибка удаления результата анализа: {e}")
            return False, f"Ошибка базы данных: {str(e)}"
        finally:
            conn.close()

