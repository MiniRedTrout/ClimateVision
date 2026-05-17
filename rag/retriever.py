import json
import math
import os
from typing import Dict, Optional, Tuple

from utils import logger


class ClimateRetriever:
    """Поиск климатических данных по локальной базе (упрощённая версия)"""

    def __init__(self, path: str = None):
        if path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(curr_dir, "knowledge_base.json")

        if not os.path.exists(path):
            logger.warning(f"Knowledge file not found: {path}")
            self.data = {}
        else:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} cities (simplified mode)")

    def find_city_by_coords(
        self, lat: float, lon: float
    ) -> Tuple[Optional[str], Optional[dict]]:
        """Находит ближайший город по координатам"""
        nearest_city = None
        nearest_data = None
        min_dist = float("inf")

        for city_key, city_data in self.data.items():
            city_lat = city_data.get("lat")
            city_lon = city_data.get("lon")
            if city_lat is None or city_lon is None:
                continue
            dist = math.sqrt((lat - city_lat) ** 2 + (lon - city_lon) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_city = city_key
                nearest_data = city_data

        if min_dist < 2.0:
            return nearest_city, nearest_data
        return None, None

    def find_city_by_name(self, city_name: str) -> Optional[Dict]:
        """Находит данные города по названию"""
        city_lower = city_name.lower()
        for city_data in self.data.values():
            if city_lower in city_data.get("city", "").lower():
                return city_data
        return None

    def get_climate_context(
        self, lat: float = None, lon: float = None, city: str = None
    ) -> str:
        """Получает климатический контекст"""
        city_data = None

        if city:
            city_data = self.find_city_by_name(city)

        if not city_data and lat and lon:
            _, city_data = self.find_city_by_coords(lat, lon)

        if not city_data:
            return ""

        return self._format_context(city_data)

    def _format_context(self, city_data: Dict) -> str:
        city_name = city_data.get("city", "Unknown")
        monthly = city_data.get("monthly", {})

        season_symbol = {"winter": "❄️", "spring": "🌸", "summer": "☀️", "autumn": "🍂"}

        context = f"\nCLIMATE KNOWLEDGE: {city_name}\n"
        context += "=" * 55 + "\n"
        context += f"{'Month':10} | {'Season':6} | {'Temp':>7} | {'Snow':>5}\n"
        context += "-" * 55 + "\n"

        for month in [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]:
            if month in monthly:
                m = monthly[month]
                symbol = season_symbol.get(m.get("season", ""), "")
                context += f"{month:10} | {symbol} {m.get('season', ''):6} | {m.get('temp', 0):6.1f}°C | {m.get('snow', 0):4.0f}mm\n"

        context += "=" * 55 + "\n"
        context += """
KEY RULES FOR SPRING vs AUTUMN:
- If the photo shows GREENING (trees getting leaves) → SPRING (March-May)
- If the photo shows YELLOWING (trees losing leaves) → AUTUMN (September-November)
- Compare the CURRENT TEMPERATURE with monthly averages above
- If current temp matches March/April/May → likely SPRING
- If current temp matches September/October/November → likely AUTUMN
"""

        return context
