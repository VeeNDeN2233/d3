"""
Универсальный обработчик файлов для всех форматов ввода Gradio.
"""

from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Обработчик видеофайлов для различных форматов ввода."""
    
    # Поддерживаемые форматы
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    # Максимальный размер файла (500 MB)
    MAX_FILE_SIZE = 500 * 1024 * 1024
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Инициализация обработчика.
        
        Args:
            temp_dir: Директория для временных файлов
        """
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "gma_videos"
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_video_path(self, input_data: Any) -> Optional[Path]:
        """
        Получить путь к видеофайлу из любого формата ввода Gradio.
        
        Args:
            input_data: Данные от Gradio File компонента
                Может быть:
                - dict: {"name": "path/to/file", ...}
                - str: путь к файлу
                - Path: объект Path
                - list: список файлов
                - None: нет файла
        
        Returns:
            Path к видеофайлу или None если файл не найден
        """
        if input_data is None:
            return None
        
        # Строка или Path
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if path.exists():
                return self._validate_and_copy(path)
            return None
        
        # Словарь (стандартный формат Gradio)
        if isinstance(input_data, dict):
            # Пробуем разные ключи, которые может использовать Gradio
            for key in ['name', 'path', 'file', 'file_path', 'file_name']:
                if key in input_data:
                    path_str = input_data[key]
                    if path_str:
                        path = Path(path_str)
                        if path.exists():
                            return self._validate_and_copy(path)
            
            # Если есть прямой путь в корне словаря
            if 'path' not in input_data and len(input_data) == 1:
                first_value = next(iter(input_data.values()))
                if isinstance(first_value, (str, Path)):
                    path = Path(first_value)
                    if path.exists():
                        return self._validate_and_copy(path)
        
        # Список файлов
        if isinstance(input_data, list):
            for item in input_data:
                path = self.get_video_path(item)
                if path:
                    return path
        
        # Объект с атрибутом name или path
        if hasattr(input_data, 'name'):
            path = Path(input_data.name)
            if path.exists():
                return self._validate_and_copy(path)
        
        if hasattr(input_data, 'path'):
            path = Path(input_data.path)
            if path.exists():
                return self._validate_and_copy(path)
        
        logger.warning(f"Не удалось извлечь путь к файлу из: {type(input_data)}")
        return None
    
    def _validate_and_copy(self, source_path: Path) -> Optional[Path]:
        """
        Валидация и копирование файла во временную директорию.
        
        Args:
            source_path: Путь к исходному файлу
        
        Returns:
            Path к скопированному файлу или None при ошибке
        """
        try:
            # Проверка существования
            if not source_path.exists():
                logger.error(f"Файл не существует: {source_path}")
                return None
            
            # Проверка расширения
            if source_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.error(f"Неподдерживаемый формат: {source_path.suffix}")
                return None
            
            # Проверка размера
            file_size = source_path.stat().st_size
            if file_size == 0:
                logger.error(f"Файл пуст: {source_path}")
                return None
            
            if file_size > self.MAX_FILE_SIZE:
                logger.error(f"Файл слишком большой: {file_size / 1024 / 1024:.2f} MB")
                return None
            
            # Копирование во временную директорию
            # Используем оригинальное имя файла для удобства
            dest_path = self.temp_dir / source_path.name
            
            # Если файл уже существует и имеет тот же размер, не копируем
            if dest_path.exists() and dest_path.stat().st_size == file_size:
                logger.info(f"Файл уже скопирован: {dest_path}")
                return dest_path
            
            shutil.copy2(source_path, dest_path)
            logger.info(f"Файл скопирован: {source_path} -> {dest_path}")
            
            return dest_path
            
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {source_path}: {e}", exc_info=True)
            return None
    
    def validate_video(self, video_path: Path) -> tuple[bool, Optional[str]]:
        """
        Валидация видеофайла.
        
        Args:
            video_path: Путь к видеофайлу
        
        Returns:
            Tuple (успех, сообщение об ошибке)
        """
        try:
            # Проверка существования
            if not video_path.exists():
                return False, "Файл не существует"
            
            # Проверка расширения
            if video_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                return False, f"Неподдерживаемый формат: {video_path.suffix}"
            
            # Проверка размера
            file_size = video_path.stat().st_size
            if file_size == 0:
                return False, "Файл пуст"
            
            if file_size > self.MAX_FILE_SIZE:
                size_mb = file_size / 1024 / 1024
                max_mb = self.MAX_FILE_SIZE / 1024 / 1024
                return False, f"Файл слишком большой: {size_mb:.2f} MB (максимум {max_mb} MB)"
            
            # Базовая проверка доступности для чтения
            try:
                with open(video_path, 'rb') as f:
                    f.read(1024)  # Читаем первые 1024 байта
            except Exception as e:
                return False, f"Файл недоступен для чтения: {str(e)}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Ошибка валидации файла: {e}", exc_info=True)
            return False, f"Ошибка валидации: {str(e)}"
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """
        Очистка старых временных файлов.
        
        Args:
            older_than_hours: Удалить файлы старше указанного количества часов
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)
            
            deleted_count = 0
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_mtime = file_path.stat().st_mtime
                    if file_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Не удалось удалить файл {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Удалено {deleted_count} временных файлов")
                
        except Exception as e:
            logger.error(f"Ошибка очистки временных файлов: {e}")

