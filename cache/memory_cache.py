import time
from typing import Any, Dict, Optional


class MemoryCache:
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, tuple] = {}
        self.ttl = ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expires_at = self._cache[key]
            if expires_at > time.time():
                self._hits += 1
                return value
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or self.ttl
        self._cache[key] = (value, time.time() + ttl)

    def delete(self, key: str):
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self):
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
            "cached_items": len(self._cache),
        }

    def cleanup_expired(self):
        now = time.time()
        expired = [k for k, (v, exp) in self._cache.items() if exp <= now]
        for key in expired:
            del self._cache[key]
        return len(expired)
