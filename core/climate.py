import aiohttp
from datetime import datetime
from cache import climate_cache, api_cache 
from utils import logger, metrics
from rag.retriever import ClimateRetriever
from omegaconf import OmegaConf
climate_retriever = ClimateRetriever()
async def get_climate_context_api(cfg:OmegaConf,lat:float,lon:float)->str:
    """Запрос к Open-Meteo API"""
    metrics.track_api_call('Open_meteo')
    try:
        url = cfg.rag.open_meteo_url if hasattr(cfg.rag, 'open_meteo_url') else "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "daily": ["temperature_2m_mean", "snowfall_sum"],
            "timezone": "auto"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
        if 'daily' not in data:
            return ""
        months_data = {}
        for i, date_str in enumerate(data["daily"]["time"]):
            month = datetime.strptime(date_str, "%Y-%m-%d").month
            temp = data["daily"]["temperature_2m_mean"][i]
            snow = data["daily"]["snowfall_sum"][i]
            if month not in months_data:
                months_data[month] = {"temps": [], "snow": 0}
            months_data[month]["temps"].append(temp)
            months_data[month]["snow"] += snow or 0
        month_names = {12: "December", 1: "January", 2: "February", 3: "March", 4: "April"}
        context = "\nClimate data for this location (based on 2023):\n"
        for month in [12, 1, 2, 3, 4]:
            if month in months_data:
                avg_temp = sum(months_data[month]["temps"]) / len(months_data[month]["temps"])
                snow = months_data[month]["snow"]
                context += f"   • {month_names[month]}: {avg_temp:.1f}°C, snow {snow:.0f}mm\n"
        return context
        
    except Exception as e:
        logger.warning(f"Climate API error: {e}")
        return ""

async def get_climate_context_hybrid(cfg:OmegaConf,lat: float=None, lon: float=None, city: str = None)->str:
    cache_key = f"climat:{lat}:{lon}:{city}"
    cached_result = climate_cache.get(cache_key)
    if cached_result:
        logger.info("Climate data from cache")
        return cached_result
    ttl = cfg.model.get('cache_ttl', 3600) if hasattr(cfg, 'model') else 86400
    if lat and lon or city:
        context = climate_retriever.get_climate_context(lat,lon,city)
        if context:
            logger.info("Using RAG")
            climate_cache.set(cache_key,context,ttl=ttl)
            return context 
    if lat and lon:
        logger.info("Using Open Meteo API")
        context = await get_climate_context_api(cfg,lat,lon)
        if context:
            climate_cache.set(cache_key,context,ttl=ttl)
            return context 
    return ''
