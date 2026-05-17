import hashlib
import json
import re
from typing import Optional, Tuple


def extract_city(caption: str) -> Optional[str]:
    """Извлечет город из подписи к картинке"""
    if not caption:
        return None
    patterns = [
        r"(?:город|в|из)\s+([А-Яа-яA-Za-z\-]+)",
        r"([А-Яа-яA-Za-z\-]+)\s+(?:город|city)",
        r"#(\w+)",
    ]
    for p in patterns:
        correct = re.search(p, caption)
        if correct:
            return correct.group(1)
    return None


def image_hash(image_path: str) -> str:
    """MD5 хэш для кэша"""
    print(f"  image_hash called with path: {image_path}", flush=True)
    with open(image_path, "rb") as f:
        data = f.read()
        print(f"  Read {len(data)} bytes", flush=True)
        result = hashlib.md5(data).hexdigest()
        print(f"  Hash result: {result}", flush=True)
        return result


def parse(txt: str) -> dict:
    """Парсим ответ от клиента"""
    txt = txt.strip()
    if txt.startswith("```json"):
        txt = txt[7:]
    if txt.startswith("```"):
        txt = txt[3:]
    if txt.endswith("```"):
        txt = txt[:-3]
    return json.loads(txt.strip())


def location(lat, lon, city, temperature) -> str:
    parts = []
    if city:
        parts.append(f"City: {city}")
    if lat and lon:
        parts.append(f"Coordinates: {lat:.4f}, {lon:.4f}")
    if temperature:
        parts.append(f"Current temperature: {temperature}°C")
    return "\n".join(parts)


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    patterns = [
        r"lat(?:itude)?\s*[:=]\s*([+-]?\d+\.?\d*)\s*[,;]\s*lon(?:gitude)?\s*[:=]\s*([+-]?\d+\.?\d*)",
        r"([+-]?\d+\.?\d*)\s*[,;]\s*([+-]?\d+\.?\d*)",
        r"([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)",
        r"lat(?:itude)?\s*-\s*([+-]?\d+\.?\d*)\s*[,;]\s*lon(?:gitude)?\s*-\s*([+-]?\d+\.?\d*)",
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
    patterns = [
        r"(?:температура|темп|temp|t)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*[cC]",
        r"([+-]?\d+(?:\.\d+)?)\s*°\s*[cC]",
        r"([+-]?\d+(?:\.\d+)?)\s*(?:градусов|градуса|град)",
    ]
    for pattern in patterns:
        match = re.search(pattern, caption, re.IGNORECASE)
        if match:
            temp = float(match.group(1))
            return temp
    return None
