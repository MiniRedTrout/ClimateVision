import aiohttp 
from utils import logger 
async def get_coordinates_by_city(city:str)->tuple:
    if not city:
        return None,None 
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
                    logger.info(f"📍 Geocoded '{city}' → {lat}, {lon}")
                    return lat, lon
    except Exception as e:
        logger.warning(f"Geocoding failed for {city}: {e}")
    
    return None, None