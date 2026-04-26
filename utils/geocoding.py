import aiohttp 
from utils import logger 
from cache import api_cache
import asyncio
async def get_coordinates_by_city(city:str)->tuple:
    if not city:
        return None,None 
    cache_key = f'geocode:{city.lower()}'
    cached = api_cache.get(cache_key)
    if cached:
        return cached 
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "SeasonBot/1.0"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()
                await asyncio.sleep(1)
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    api_cache.set(cache_key, (lat, lon), ttl=604800)
                    logger.info(f"📍 Geocoded '{city}' → {lat}, {lon}")
                    return lat, lon
    except Exception as e:
        logger.warning(f"Geocoding failed for {city}: {e}")
    
    return None, None