import re 
import hashlib
from typing import Optional, Tuple
from pathlib import Path 
import json
def extract_city(caption:str)->Optional[str]:
    """Извлечет город из подписи к картинке"""
    if not caption:
        return None 
    patterns = [
        r'(?:город|в|из)\s+([А-Яа-яA-Za-z\-]+)',
        r'([А-Яа-яA-Za-z\-]+)\s+(?:город|city)',
        r'#(\w+)'
    ]
    for p in patterns:
        correct = re.search(p,caption)
        if correct:
            return correct.group(1)
        return None 
def image_hash(image_path: str)->str:
    """MD5 хэш для кэша"""
    with open(image_path,'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
def parse(txt: str)->dict:
    """Парсим ответ от клиента"""
    txt = txt.strip()
    if txt.startswith('```json'):
        txt = txt[7:]
    if txt.startswith('```'):
        txt = txt[3:]
    if txt.endswith('```'):
        txt = txt[:-3]
    return json.loads(txt.strip())
def location(lat: Optional[float], lon: Optional[float], city: Optional[str])->str:
    """По красоте в промпт"""
    if city:
        return f'Locatio: {city}'
    elif lat and lon:
        return f'Location: {lat:.4f}, {lon:.4f}'
    return ""
