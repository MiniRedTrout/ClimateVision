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
    print(f"  image_hash called with path: {image_path}", flush=True)
    with open(image_path,'rb') as f:
        data = f.read()
        print(f"  Read {len(data)} bytes", flush=True)
        result = hashlib.md5(data).hexdigest()
        print(f"  Hash result: {result}", flush=True)
        return result
    
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
def location(lat: Optional[float], lon: Optional[float], city: Optional[str], temperature: Optional[str])->str:
    """По красоте в промпт"""
    prompt = ''
    if city:
        prompt += f'Location: {city}'
    if lat and lon:
        prompt += f'Location: {lat:.4f}, {lon:.4f}'
    if temperature:
        prompt += f'Temperature: {temperature}'
    return prompt
def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    patterns = [
        r'lat(?:itude)?\s*[:=]\s*([+-]?\d+\.?\d*)\s*[,;]\s*lon(?:gitude)?\s*[:=]\s*([+-]?\d+\.?\d*)',
        r'([+-]?\d+\.?\d*)\s*[,;]\s*([+-]?\d+\.?\d*)',
        r'([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)',
        r'lat(?:itude)?\s*-\s*([+-]?\d+\.?\d*)\s*[,;]\s*lon(?:gitude)?\s*-\s*([+-]?\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return None

def extract_temperature(caption: str) -> Optional[float]:
    if not caption:
        return None
    patterns = r'(?:temp|temperature|t)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)'
    
    for pattern in patterns:
        match = re.search(pattern, caption, re.IGNORECASE)
        if match:
            try:
                temp = float(match.group(1))
                if -50 <= temp <= 60:
                    return temp
            except ValueError:
                continue
    return None

def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    patterns = [
        r'\{[^{}]*"season"[^{}]*"month"[^{}]*"confidence"[^{}]*\}',
        r'\{[^{}]*"season"[^{}]*"month"[^{}]*\}',
        r'\{[^{}]*"season"[^{}]*\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group())
            except:
                continue
    
    return None