import json
import math
import os
from typing import Dict, Optional, Tuple

from utils import logger

# Маппинг русских названий → ключи в knowledge_base.json
CITY_ALIASES = {
    # А
    "абакан": "abakan",
    "анапа": "anapa",
    "архангельск": "arkhangelsk",
    "астрахань": "astrakhan",
    # Б
    "барнаул": "barnaul",
    "белгород": "belgorod",
    "брянск": "bryansk",
    # В
    "владикавказ": "vladikavkaz",
    "владимир": "vladimir",
    "волгоград": "volgograd",
    "вологда": "vologda",
    "воронеж": "voronezh",
    # Г
    "геленджик": "gelendzhik",
    "грозный": "grozny",
    # Д
    # Е
    "екатеринбург": "ekaterinburg",
    # И
    "иркутск": "irkutsk",
    "иваново": "ivanovo",
    "ижевск": "izhevsk",
    # К
    "казань": "kazan",
    "калининград": "kaliningrad",
    "калуга": "kaluga",
    "кемерово": "kemerovo",
    "хабаровск": "khabarovsk",
    "ханты-мансийск": "khanty_mansiysk",
    "киров": "kirov",
    "краснодар": "krasnodar",
    "красноярск": "krasnoyarsk",
    "курган": "kurgan",
    "курск": "kursk",
    # Л
    "липецк": "lipetsk",
    # М
    "магнитогорск": "magnitogorsk",
    "махачкала": "makhachkala",
    "минеральные воды": "mineralnye_vody",
    "минводы": "mineralnye_vody",
    "москва": "moscow",
    "мск": "moscow",
    "мурманск": "murmansk",
    # Н
    "нальчик": "nalchik",
    "назрань": "nazran",
    "нерюнгри": "neryungri",
    "нижневартовск": "nizhnevartovsk",
    "нижний новгород": "nizhny_novgorod",
    "новокузнецк": "novokuznetsk",
    "новороссийск": "novorossiysk",
    "новосибирск": "novosibirsk",
    # О
    "омск": "omsk",
    "орел": "orel",
    "орёл": "orel",
    "оренбург": "orenburg",
    # П
    "пенза": "penza",
    "пермь": "perm",
    "петропавловск-камчатский": "petropavlovsk_kamchatsky",
    "петропавловск камчатский": "petropavlovsk_kamchatsky",
    "псков": "pskov",
    "пятигорск": "pyatigorsk",
    # Р
    "ростов": "rostov",
    "ростов-на-дону": "rostov",
    "ростов на дону": "rostov",
    "рязань": "ryazan",
    # С
    "санкт-петербург": "saint_petersburg",
    "петербург": "saint_petersburg",
    "питер": "saint_petersburg",
    "спб": "saint_petersburg",
    "самара": "samara",
    "саранск": "saransk",
    "саратов": "saratov",
    "смоленск": "smolensk",
    "сочи": "sochi",
    "ставрополь": "stavropol",
    "сургут": "surgut",
    "сыктывкар": "syktyvkar",
    # Т
    "тамбов": "tambov",
    "томск": "tomsk",
    "тула": "tula",
    "тверь": "tver",
    "тюмень": "tyumen",
    # У
    "уфа": "ufa",
    "улан-удэ": "ulan_ude",
    "улан удэ": "ulan_ude",
    "ульяновск": "ulyanovsk",
    # Ч
    "чебоксары": "cheboksary",
    "челябинск": "chelyabinsk",
    "чита": "chita",
    # Я
    "якутск": "yakutsk",
    "ярославль": "yaroslavl",
    # Страны СНГ / мир — популярные
    "астана": "astana",
    "баку": "baku",
    "бишкек": "bishkek",
    "ереван": "yerevan",
    "ерева́н": "yerevan",
    "минск": "minsk",
    "киев": "kyiv",
    "ташкент": "tashkent",
    "душанбе": "dushanbe",
    "ашхабад": "ashgabat",
    "пекин": "beijing",
    "берлин": "berlin",
    "париж": "paris",
    "лондон": "london",
    "рим": "rome",
    "мадрид": "madrid",
    "стамбул": "istanbul",
    "токио": "tokyo",
    "бангкок": "bangkok",
    "дубай": "dubai",
    "варшава": "warsaw",
    "прага": "prague",
    "будапешт": "budapest",
    "белград": "belgrade",
    "бухарест": "bucharest",
    "софия": "sofia",
    "хельсинки": "helsinki",
    "стокгольм": "stockholm",
    "осло": "oslo",
    "копенгаген": "copenhagen",
    "вена": "vienna",
    "анkara": "ankara",
    "каир": "cairo",
    "делхи": "delhi",
    "дели": "delhi",
}


class ClimateRetriever:
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

        if min_dist < 2.0:  # ~200 км
            return nearest_city, nearest_data
        return None, None

    def find_city_by_name(self, city_name: str) -> Optional[Dict]:
        city_lower = city_name.lower().strip()

        # 1. Проверяем русские алиасы
        mapped_key = CITY_ALIASES.get(city_lower)
        if mapped_key and mapped_key in self.data:
            return self.data[mapped_key]

        # 2. Поиск по ключам в базе (английские lowercase)
        if city_lower in self.data:
            return self.data[city_lower]

        # 3. Частичное совпадение с полем "city"
        for city_key, city_data in self.data.items():
            if city_lower in city_data.get("city", "").lower():
                return city_data
            if city_lower == city_key.lower():
                return city_data

        return None

    def get_climate_context(
        self, lat: float = None, lon: float = None, city: str = None
    ) -> str:
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

        context = f"\n🏙️ CLIMATE KNOWLEDGE: {city_name}\n"
        context += "=" * 50 + "\n"

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
                season_symbol = {
                    "winter": "❄️",
                    "spring": "🌸",
                    "summer": "☀️",
                    "autumn": "🍂",
                }.get(m.get("season", ""), "")
                context += f"{month:10} | {season_symbol} {m.get('season', ''):6} | {m.get('temp', 0):5.1f}°C | snow: {m.get('snow', 0):3.0f}mm\n"

        context += "=" * 50 + "\n"
        context += "RULE: Use this climate data as PRIMARY reference. Compare user's temperature with monthly averages.\n"

        return context
