import hashlib
import json
import re
from typing import Optional, Tuple


def extract_city(caption: str) -> Optional[str]:
    """Извлекает город из подписи к картинке"""
    if not caption:
        return None

    patterns = [
        # "город Москва", "city Moscow"
        r"(?:город|city)\s+([А-ЯЁа-яёA-Za-z\-\s]+?)(?:\s*[,.]|\s*$)",
        # "в Москве", "в Moscow", "из Москвы"
        r"(?:в|из|near|from|at)\s+([А-ЯЁа-яёA-Za-z\-]{3,})",
        # "#москва"
        r"#([а-яёa-z]{3,})",
        # "Москва, март" — город перед запятой
        r"^([А-ЯЁ][а-яёA-Za-z\-]{2,})\s*,",
        # "city/town/сити/город: Moscow"
        r"(?:city|town|сити|город)\s*[:=]?\s*([А-ЯЁA-Z][а-яёa-z\-]{2,})",
    ]

    garbage = {
        "январь",
        "февраль",
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
        "весна",
        "лето",
        "осень",
        "зима",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "spring",
        "summer",
        "autumn",
        "winter",
        "temp",
        "temperature",
        "температура",
        "градусов",
        "градуса",
        "град",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "широта",
        "долгота",
        "north",
        "south",
        "east",
        "west",
        "север",
        "юг",
        "восток",
        "запад",
        "n",
        "s",
        "e",
        "w",
    }

    for p in patterns:
        match = re.search(p, caption, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            if city.lower() not in garbage and len(city) >= 2:
                return city

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
    """Парсим ответ от модели — выковыривает JSON из любого мусора"""
    txt = txt.strip()

    # Убираем markdown обёртки
    if txt.startswith("```json"):
        txt = txt[7:]
    if txt.startswith("```"):
        txt = txt[3:]
    if txt.endswith("```"):
        txt = txt[:-3]
    txt = txt.strip()

    # Попытка 1: прямой парсинг
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass

    # Попытка 2: найти JSON-объект по скобкам
    brace_start = txt.find("{")
    brace_end = txt.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(txt[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Попытка 3: regex по ключам (финальный fallback)
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
    """Формирует строку с локацией для промпта"""
    parts = []
    if city:
        parts.append(f"City: {city}")
    if lat and lon:
        parts.append(f"Coordinates: {lat:.4f}, {lon:.4f}")
    if temperature:
        parts.append(f"Current temperature: {temperature}°C")
    return "\n".join(parts)


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """Парсит координаты из текста — поддерживает множество форматов"""
    if not text:
        return None

    patterns_lat_lon = [
        # lat=55.75, lon=37.62 / lat:55.75; lon:37.62
        r"lat(?:itude)?\s*[:=]?\s*([+-]?\d+\.?\d*)\s*[,;\s]\s*lon(?:g(?:itude)?)?\s*[:=]?\s*([+-]?\d+\.?\d*)",
        # lon first, then lat
        r"lon(?:g(?:itude)?)?\s*[:=]?\s*([+-]?\d+\.?\d*)\s*[,;\s]\s*lat(?:itude)?\s*[:=]?\s*([+-]?\d+\.?\d*)",
        # кириллица: ш/шр/широта ... д/дг/долгота
        r"(?:ш|ш\.|шр)\s*[:=]?\s*([+-]?\d+\.?\d*)\s*[,;\s]\s*(?:д|д\.|дг)\s*[:=]?\s*([+-]?\d+\.?\d*)",
        r"широта\s*[:=]?\s*([+-]?\d+\.?\d*)\s*[,;\s]\s*долгота\s*[:=]?\s*([+-]?\d+\.?\d*)",
        # компас: 55°N 37°E (с градусом и без)
        r"([\d]+\.?\d*)\s*°?\s*[NnСс]\s*[,;\s]+\s*([\d]+\.?\d*)\s*°?\s*[EeВв]",
        r"([\d]+\.?\d*)\s*°?\s*[SsЮю]\s*[,;\s]+\s*([\d]+\.?\d*)\s*°?\s*[EeВв]",
        r"([\d]+\.?\d*)\s*°?\s*[NnСс]\s*[,;\s]+\s*([\d]+\.?\d*)\s*°?\s*[WwЗз]",
        r"([\d]+\.?\d*)\s*°?\s*[SsЮю]\s*[,;\s]+\s*([\d]+\.?\d*)\s*°?\s*[WwЗз]",
        r"([\d]+\.?\d*)\s*[NnСс]\s*[,;\s]+\s*([\d]+\.?\d*)\s*[EeВв]",
        r"([\d]+\.?\d*)\s*[NnСс]\s*[,;\s]+\s*([\d]+\.?\d*)\s*[WwЗз]",
        r"([\d]+\.?\d*)\s*[SsЮю]\s*[,;\s]+\s*([\d]+\.?\d*)\s*[EeВв]",
        r"([\d]+\.?\d*)\s*[SsЮю]\s*[,;\s]+\s*([\d]+\.?\d*)\s*[WwЗз]",
    ]

    for pattern in patterns_lat_lon:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val1 = float(match.group(1))
                val2 = float(match.group(2))

                # Определяем порядок lat/lon и знак
                if "lon" in pattern and "lat" in pattern:
                    if pattern.index("lon") < pattern.index("lat"):
                        lat, lon = val2, val1
                    else:
                        lat, lon = val1, val2
                elif re.search(r"[SsЮю]", pattern, re.IGNORECASE):
                    lat = -abs(val1)
                    lon = val2
                elif re.search(r"[WwЗз]", match.group(0), re.IGNORECASE):
                    lat = val1
                    lon = -abs(val2)
                else:
                    lat = val1
                    lon = val2

                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except ValueError:
                continue

    # Fallback: просто два числа подряд
    fallback = re.findall(r"([+-]?\d{1,3}\.?\d*)", text)
    if len(fallback) >= 2:
        try:
            val1 = float(fallback[0])
            val2 = float(fallback[1])
            if -90 <= val1 <= 90 and -180 <= val2 <= 180:
                return val1, val2
        except ValueError:
            pass

    return None


def extract_temperature(caption: str) -> Optional[float]:
    """Извлекает температуру из подписи — поддерживает множество форматов"""
    if not caption:
        return None

    patterns = [
        # "температура: 5", "темп = 5", "temp: 5", "Т: 5"
        r"(?:температура|темп|temp|temperature|t|Т)\s*[:=]\s*([+-]?\d+(?:[.,]\d+)?)",
        # "температура 5", "temp 5"
        r"(?:температура|темп|temp|temperature|t|Т)\s+([+-]?\d+(?:[.,]\d+)?)",
        # "5°C", "+5°C", "-10°C"
        r"([+-]?\d+(?:[.,]\d+)?)\s*°\s*[cCСс]",
        # "5°" (без буквы)
        r"([+-]?\d+(?:[.,]\d+)?)\s*°",
        # "5 градусов", "+5 градусов", "-10 градуса"
        r"([+-]?\d+(?:[.,]\d+)?)\s*(?:градусов|градуса|град)",
        # "плюс 5", "+ 5"
        r"(?:плюс|\+)\s*(\d+(?:[.,]\d+)?)\s*(?:градусов|градуса|°)?",
        # "минус 10", "− 10", "— 10"
        r"(?:минус|−|—)\s*(\d+(?:[.,]\d+)?)\s*(?:градусов|градуса|°)?",
        # "5 градусов тепла", "5 выше нуля"
        r"(\d+(?:[.,]\d+)?)\s*(?:градусов\s+тепла|выше\s+нуля)",
        # "5 градусов холода/мороза", "5 ниже нуля"
        r"(\d+(?:[.,]\d+)?)\s*(?:градусов\s+(?:холода|мороза)|ниже\s+нуля)",
        # "5с" (без пробела)
        r"(\d+(?:[.,]\d+)?)\s*[cCСс]\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, caption, re.IGNORECASE)
        if match:
            try:
                temp_str = match.group(1).replace(",", ".")
                temp = float(temp_str)

                # Определяем знак по контексту
                full_match = match.group(0).lower()
                if any(
                    w in full_match
                    for w in ["холода", "мороза", "ниже нуля", "минус", "−", "—"]
                ):
                    temp = -abs(temp)
                elif any(w in full_match for w in ["плюс", "выше нуля", "тепла"]):
                    temp = abs(temp)

                if -60 <= temp <= 60:
                    return temp
            except ValueError:
                continue
    return None


def extract_json_from_response(text: str) -> dict:
    """Извлекает JSON из ответа модели (алиас для parse с доп. regex-паттернами)"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, Exception):
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
            except (json.JSONDecodeError, Exception):
                continue

    return None
