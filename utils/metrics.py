import time 
from functools import wraps 
from typing import Dict, Any 
from collections import defaultdict
class Metrics:
    """Собираем метрики"""
    def __init__(self):
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = defaultdict(int)
        self.responses_times = []
    def track_request(self):
        self.request_count += 1
    def track_cache_hit(self):
        self.cache_hits += 1
    def track_cache_miss(self):
        self.cache_misses += 1
    def track_api_call(self, api_name:str):
        self.api_calls[api_name] += 1
    def track_response_time(self, resp_time: float):
        self.responses_times.append(resp_time)
    def get_stats(self)->Dict[str,Any]:
        avg_time = sum(self.responses_times)/len(self.responses_times) if self.responses_times else 0
        cache_hit_rate = self.cache_hits /(self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        return {
            'requests': self.request_count,
            'cache_hits':self.cache_hits,
            'cache_misses':self.cache_misses,
            'cache_hit_rate':cache_hit_rate,
            'avg_response_time':avg_time,
            'api_calls':dict(self.api_calls)
        }
    def reset(self):
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls.clear()
        self.responses_times.clear()

def track_time(func):
    """Декоратор для времени выполнения"""
    @wraps(func)
    async def wrapper(*args,**kwargs):
        start = time.time()
        result = await func(*args,**kwargs)
        dur = (time.time() - start)*1000
        metrics.track_response_time(dur)
        return result
    return wrapper
metrics = Metrics