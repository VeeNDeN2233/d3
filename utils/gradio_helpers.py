"""
Вспомогательные функции для работы с Gradio компонентами.
"""

from typing import Any, Optional, Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_status_message(message: str, status_type: str = "info") -> str:
    """
    Создать HTML сообщение статуса.
    
    Args:
        message: Текст сообщения
        status_type: Тип сообщения (info, success, error, warning)
    
    Returns:
        HTML строка
    """
    status_classes = {
        "info": "status-info",
        "success": "status-success",
        "error": "status-error",
        "warning": "status-warning",
    }
    
    css_class = status_classes.get(status_type, "status-info")
    return f"<div class='{css_class}'>{message}</div>"


def format_file_size(size_bytes: int) -> str:
    """
    Форматировать размер файла в читаемый вид.
    
    Args:
        size_bytes: Размер в байтах
    
    Returns:
        Отформатированная строка
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def create_progress_html(progress: float, message: str) -> str:
    """
    Создать HTML для индикатора прогресса.
    
    Args:
        progress: Прогресс от 0 до 1
        message: Сообщение о текущем этапе
    
    Returns:
        HTML строка
    """
    progress_percent = int(progress * 100)
    return f"""
    <div class="progress-container" style="margin: 20px 0;">
        <div class="progress-bar" style="width: 100%; height: 24px; background: #e2e8f0; border-radius: 12px; overflow: hidden; position: relative;">
            <div class="progress-fill" style="width: {progress_percent}%; height: 100%; background: linear-gradient(90deg, #4a90e2 0%, #357abd 100%); transition: width 0.3s ease;"></div>
            <div class="progress-text" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: 600; color: #2d3748;">{progress_percent}%</div>
        </div>
        <p style="margin: 8px 0 0 0; font-size: 14px; color: #4a5568; text-align: center;">{message}</p>
    </div>
    """

