import os 
import json 
import time 
import hashlib 
from pathlib import Path
from typing import Any, Optional
from utils import logger 

class FileCache:
    """Кэш"""
    def __init__(self,cache_dir: str = "cache_files"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._hits = 0
        self._misses = 0
    def _get_path(self,key:str)->Path:
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.json"
    def get(self,key:str)->Optional[Any]:
        path = self._get_path(key)
        if path.exists():
            try:
                with open(path,'r',encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('expires_at', 0) > time.time():
                        self._hits += 1
                        return data.get('value')
                    else:
                        path.unlink()
            except Exception as e:
                logger.warning(f'Failde to read cache: {e}')
        self._misses += 1
        return None 
    def set(self, key: str, value: Any, ttl: int = 3600):
        path = self._get_path(key)
        try:
            with open(path, 'w',encoding='utf-8') as f:
                json.dump({
                    'value': value,
                    'expires_at':time.time() + ttl
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f'Failed to write cache: {e}')
    def delete(self, key: str):
        path = self._get_path(key)
        if path.exists():
            path.unlink()
    def clear(self):
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
        self._hits = 0
        self._misses = 0
    def get_stats(self):
        total = self._misses + self._hits 
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses':self._misses,
            'total':total,
            'hit_rate':hit_rate
        }
    
