import time 
from collections import defaultdict
from typing import Dict, Tuple
from utils import logger 
from omegaconf import DictConfig

class RateLimiter:
    """Ограничивает частоту запросов"""
    def __init__(self,cfg: DictConfig):
        self._requests: Dict[int,list] = defaultdict(list)
        self.cfg = cfg
    def is_allowed(self,user_id:int)->Tuple[bool,int]:
        now = time.time()
        window_start = now - self.cfg.middleware.rate_seconds
        self._requests[user_id] = [
            req_time for req_time in self._requests[user_id]
            if req_time > window_start
        ]
        if len(self._requests[user_id]) >=self.cfg.rate_limiter.requests_per_minute:
            old = min(self._requests[user_id])
            wait_time = int(self.cfg.rate_limiter.seconds - (now - old))
            return False, max(1, wait_time)
        self._requests[user_id].append(now)
        return True, 0
    def reset_user(self, user_id:int):
        if user_id in self._requests:
            del self._requests[user_id]
    def get_stats(self, user_id:int)->Dict:
        now = time.time()
        window_start = now - self.cfg.rate_limiter.seconds
        recent = [
            req_time for req_time in self._requests.get(user_id,[])
            if req_time > window_start
        ]
        return {
            'user_id':user_id,
            'requests_in_window':len(recent)
        }

rate_limiter = RateLimiter()