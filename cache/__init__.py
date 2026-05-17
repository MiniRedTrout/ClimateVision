from .file_cache import FileCache
from .memory_cache import MemoryCache

ollama_cache = MemoryCache(ttl=3600)
climate_cache = MemoryCache(ttl=86400)
api_cache = MemoryCache(ttl=604800)
file_cache = FileCache()
