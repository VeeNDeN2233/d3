"""
Скрипт для создания администратора системы.
"""

import sys
import os
from pathlib import Path
from auth.database import UserDatabase
from auth.auth_manager import AuthManager

# Устанавливаем UTF-8 для вывода
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def create_admin():
    """Создать администратора с email admin@admin.com."""
    
    db = UserDatabase()
    auth_manager = AuthManager()
    
    admin_email = "admin@admin.com"
    admin_username = "admin"
    admin_password = "admin123"  # В продакшене использовать более сложный пароль
    
    print(f"Создание администратора...")
    print(f"Email: {admin_email}")
    print(f"Username: {admin_username}")
    
    # Проверяем, существует ли уже админ
    import sqlite3
    conn_obj = sqlite3.connect(str(db.db_path))
    cursor = conn_obj.cursor()
    
    cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (admin_email, admin_username))
    existing = cursor.fetchone()
    
    if existing:
        print("WARNING: Администратор уже существует!")
        print("Пересоздаем администратора...")
        
        # Удаляем существующего админа
        admin_id = existing[0]
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (admin_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (admin_id,))
        conn_obj.commit()
        print("Старый администратор удален.")
    
    conn_obj.close()
    
    # Создаем нового админа
    success, message = db.create_user(
        username=admin_username,
        password=admin_password,
        email=admin_email,
        full_name="System Administrator",
        role="admin"
    )
    
    if success:
        print(f"SUCCESS: {message}")
        print(f"\nДанные для входа:")
        print(f"  Email: {admin_email}")
        print(f"  Username: {admin_username}")
        print(f"  Password: {admin_password}")
        print(f"\nВАЖНО: Измените пароль после первого входа!")
    else:
        print(f"ERROR: {message}")
        sys.exit(1)


if __name__ == "__main__":
    create_admin()
