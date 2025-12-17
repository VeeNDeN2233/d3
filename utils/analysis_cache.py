
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AnalysisCache:
    
    def __init__(self, cache_dir: Path = Path("cache/analysis")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = 7
    
    def _get_cache_key(self, video_path: Path, age_weeks: int, gestational_age: int) -> str:

        stat = video_path.stat()
        key_data = f"{video_path}_{stat.st_size}_{stat.st_mtime}_{age_weeks}_{gestational_age}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, video_path: Path, age_weeks: int, gestational_age: int) -> Optional[Dict[str, Any]]:
        try:
            cache_key = self._get_cache_key(video_path, age_weeks, gestational_age)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if not cache_file.exists():
                return None
            

            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > timedelta(days=self.max_age_days):
                logger.info(f"Кэш устарел, удаляем: {cache_file}")
                cache_file.unlink()
                return None
            

            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            

            if 'video_stat' in data:
                current_stat = video_path.stat()
                cached_stat = data['video_stat']
                if (current_stat.st_size != cached_stat.get('size') or
                    current_stat.st_mtime != cached_stat.get('mtime')):
                    logger.info("Исходный файл изменился, кэш недействителен")
                    cache_file.unlink()
                    return None
            
            logger.info(f"Результаты загружены из кэша: {cache_file}")
            return data.get('results')
            
        except Exception as e:
            logger.error(f"Ошибка при чтении кэша: {e}", exc_info=True)
            return None
    
    def set(
        self,
        video_path: Path,
        age_weeks: int,
        gestational_age: int,
        results: Dict[str, Any]
    ) -> bool:
        try:
            cache_key = self._get_cache_key(video_path, age_weeks, gestational_age)
            cache_file = self.cache_dir / f"{cache_key}.json"
            

            stat = video_path.stat()
            cache_data = {
                'results': results,
                'video_path': str(video_path),
                'video_stat': {
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                },
                'parameters': {
                    'age_weeks': age_weeks,
                    'gestational_age': gestational_age,
                },
                'cached_at': datetime.now().isoformat(),
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Результаты сохранены в кэш: {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении в кэш: {e}", exc_info=True)
            return False
    
    def cleanup_old(self):
        try:
            deleted_count = 0
            cutoff_time = datetime.now() - timedelta(days=self.max_age_days)
            
            for cache_file in self.cache_dir.glob("*.json"):
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    cache_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Удалено {deleted_count} устаревших записей кэша")
                
        except Exception as e:
            logger.error(f"Ошибка при очистке кэша: {e}", exc_info=True)

