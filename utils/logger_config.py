"""
Конфигурация системы логирования для админки.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from logging.handlers import RotatingFileHandler

# Создаем директорию для логов
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Путь к файлу логов
LOG_FILE = LOG_DIR / "app.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"
ADMIN_LOG_FILE = LOG_DIR / "admin.log"


def setup_logging():
    """Настройка системы логирования."""
    
    # Формат логов
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Основной логгер приложения
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Удаляем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # Обработчик для основного файла логов (ротация при 10MB, максимум 5 файлов)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Обработчик для ошибок (только ERROR и выше)
    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    root_logger.addHandler(error_handler)
    
    # Специальный логгер для админки
    admin_logger = logging.getLogger('admin')
    admin_logger.setLevel(logging.INFO)
    
    admin_handler = RotatingFileHandler(
        ADMIN_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding='utf-8'
    )
    admin_handler.setLevel(logging.INFO)
    admin_handler.setFormatter(log_format)
    admin_logger.addHandler(admin_handler)
    
    # Предотвращаем дублирование логов
    admin_logger.propagate = False
    
    return admin_logger


def get_log_entries(log_file: Path, lines: int = 100) -> List[Dict]:
    """
    Получить последние записи из лог-файла.
    
    Args:
        log_file: Путь к файлу логов
        lines: Количество строк для чтения
    
    Returns:
        Список словарей с записями логов
    """
    entries = []
    
    if not log_file.exists():
        return entries
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            # Читаем последние N строк
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Парсим формат: "2024-12-17 10:30:45 - logger_name - LEVEL - message"
                try:
                    parts = line.split(' - ', 3)
                    if len(parts) >= 4:
                        timestamp = parts[0]
                        logger_name = parts[1]
                        level = parts[2]
                        message = parts[3]
                        
                        entries.append({
                            'timestamp': timestamp,
                            'logger': logger_name,
                            'level': level,
                            'message': message,
                            'raw': line
                        })
                    else:
                        entries.append({
                            'timestamp': '',
                            'logger': '',
                            'level': 'INFO',
                            'message': line,
                            'raw': line
                        })
                except Exception as e:
                    entries.append({
                        'timestamp': '',
                        'logger': '',
                        'level': 'INFO',
                        'message': line,
                        'raw': line
                    })
    except Exception as e:
        logging.error(f"Ошибка чтения лог-файла {log_file}: {e}")
    
    return entries


# Инициализация логирования при импорте
admin_logger = setup_logging()
