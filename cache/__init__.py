from .memory_cache import MemoryCache
from .file_cache import FileCache

ollama_cache = MemoryCache()
climate_cache = MemoryCache()
api_cache = MemoryCache()
file_cache = FileCache()