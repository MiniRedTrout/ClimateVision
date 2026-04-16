import aiohttp
import asyncio
from typing import Optional, Tuple

async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 10
            }
            headers = {
                "User-Agent": "SeasonBot/1.0"
            }
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    city = data.get('address', {}).get('city') or \
                           data.get('address', {}).get('town') or \
                           data.get('address', {}).get('village')
                    return city
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
    return None

async def check_ollama_health(host: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{host}/api/tags") as resp:
                return resp.status == 200
    except:
        return False