import hashlib
import json
import re
from typing import Optional, Tuple


def extract_city(caption: str) -> Optional[str]:
    if not caption:
        return None

    patterns = [
        r"(?:город|city)\s+([А-ЯЁа-яёA-Za-z\-\s]+?)(?:\s*[,.]|\s*$)",
        r"(?:в|из|near|from|at)\s+([А-ЯЁа-яёA-Za-z\-]{3,})",
        r"#([а-яёa-z]{3,})",
        r"^([А-ЯЁ][а-яёA-Za-z\-]{2,})\s*,",
        r"(?:city|town|сити|город)\s*[:=]?\s*([А-ЯЁA-Z][а-яёa-z\-]{2,})",
    ]

    for p in patterns:
        match = re.search(p, caption, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            garbage_words = [
                "март",
                "апрель",
                "май",
                "июнь",
                "июль",
                "август",
                "сентябрь",
                "октябрь",
                "ноябрь",
                "декабрь",
                "январь",
                "февраль",
                "весна",
                "лето",
                "осень",
                "зима",
                "spring",
                "summer",
                "autumn",
                "winter",
                "temp",
                "температура",
                "градусов",
            ]
            if city.lower() not in garbage_words and len(city) >= 2:
                return city

    return None


def image_hash(image_path: str) -> str:
    print(f"  image_hash called with path: {image_path}", flush=True)
    with open(image_path, "rb") as f:
        data = f.read()
        print(f"  Read {len(data)} bytes", flush=True)
        result = hashlib.md5(data).hexdigest()
        print(f"  Hash result: {result}", flush=True)
        return result


def parse(txt: str) -> dict:
    txt = txt.strip()

    if txt.startswith("```json"):
        txt = txt[7:]
    if txt.startswith("```"):
        txt = txt[3:]
    if txt.endswith("```"):
        txt = txt[:-3]
    txt = txt.strip()

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass

    brace_start = txt.find("{")
    brace_end = txt.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(txt[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    season_match = re.search(r'"season"\s*:\s*"([^"]+)"', txt)
    month_match = re.search(r'"month"\s*:\s*"([^"]+)"', txt)
    confidence_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', txt)

    result = {}
    if season_match:
        result["season"] = season_match.group(1)
    if month_match:
        result["month"] = month_match.group(1)
    if confidence_match:
        result["confidence"] = confidence_match.group(1)

    if result:
        if "confidence" not in result:
            result["confidence"] = "medium"
        return result

    return {"season": "unknown", "month": "unknown", "confidence": "low"}


def location(
    lat: Optional[float],
    lon: Optional[float],
    city: Optional[str],
    temperature: Optional[str],
) -> str:
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
        r"([+-]?\d{1,3}\.?\d*)\s*[,;]\s*([+-]?\d{1,3}\.?\d*)",
        r"([+-]?\d{1,2}\.?\d*)\s+([+-]?\d{1,3}\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except ValueError:
                continue
    return None


def extract_temperature(caption: str) -> Optional[float]:
    if not caption:
        return None

    patterns = [
        r"(?:температура|темп|temp|t)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*[cC]?",
        r"([+-]?\d+(?:\.\d+)?)\s*°\s*[cC]",
        r"([+-]?\d+(?:\.\d+)?)\s*(?:градусов|градуса|град)",
        r"(?:плюс|\+)\s*(\d+(?:\.\d+)?)",
        r"(?:минус|\-)\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:градусов\s+тепла)",
        r"(\d+(?:\.\d+)?)\s*(?:градусов\s+холода)",
    ]

    for pattern in patterns:
        match = re.search(pattern, caption, re.IGNORECASE)
        if match:
            try:
                temp = float(match.group(1))
                if "холода" in match.group(0) or "минус" in match.group(0).lower():
                    temp = -abs(temp)
                return temp
            except ValueError:
                continue
    return None
