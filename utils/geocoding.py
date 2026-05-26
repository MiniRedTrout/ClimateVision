import aiohttp

from cache import api_cache
from utils import logger


async def get_coordinates_by_city(city: str) -> tuple:
    if not city:
        return None, None

    cache_key = f"geocode:{city.lower()}"
    cached = api_cache.get(cache_key)
    if cached:
        return cached

    lat, lon = await _openmeteo_geocode(city)
    if lat:
        api_cache.set(cache_key, (lat, lon), ttl=604800)
        return lat, lon

    lat, lon = await _nominatim_geocode(city)
    if lat:
        api_cache.set(cache_key, (lat, lon), ttl=604800)
        return lat, lon

    return None, None


async def _openmeteo_geocode(city: str) -> tuple:
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            async with session.get(
                url, params={"name": city, "count": 1, "language": "en"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("results"):
                        r = data["results"][0]
                        lat = r["latitude"]
                        lon = r["longitude"]
                        logger.info(f"Open-Meteo geocoded '{city}' -> {lat}, {lon}")
                        return lat, lon
    except Exception as e:
        logger.warning(f"Open-Meteo geocoding error: {e}")
    return None, None


async def _nominatim_geocode(city: str) -> tuple:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "SeasonBot/1.0"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
                        logger.info(f"Nominatim geocoded '{city}' -> {lat}, {lon}")
                        return lat, lon
                elif resp.status == 429:
                    logger.warning(f"Nominatim rate limit for {city}")
    except Exception as e:
        logger.warning(f"Nominatim geocoding error: {e}")
    return None, None
