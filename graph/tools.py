from typing import Optional
from utils.logger import logger 
from langchain_core.tools import tool
import json
from core.mcp_client import OpenMeteoMCPClient

def get_openmeteo_client():
    global _openmeteo_client 
    if _openmeteo_client is None:
        _openmeteo_client = OpenMeteoMCPClient()
    return _openmeteo_client

@tool 
async def get_climate_history(lat: float, lon: float, year: int = 2023)->str:
    """
    Получить исторические климатические данные для указанных координат.
    Используй когда нужно узнать о климате, погоде за прошлые периоды.
    
    Args:
        lat: широта (от -90 до 90)
        lon: долгота (от -180 до 180)
        year: год для получения исторических данных (по умолчанию 2023)
    
    Returns:
        JSON строка с историческими данными о температуре, осадках и ветре
    """
    try:
        client = get_openmeteo_client()
        climate_data = await client.get_climate_history(lat, lon, year)
        return climate_data
    except Exception as e:
        logger.error(f"Error in get_climate_history: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@tool
async def get_seasonal_forecast(lat: float, lon: float) -> str:
    """
    Получить сезонный прогноз на 7 месяцев вперед (SEAS5 модель).
    Используй когда нужно понять ожидаемые сезонные условия.
    """
    client = get_openmeteo_client()
    return await client.get_seasonal_forecast(lat, lon)

@tool
async def get_climate_normals(lat: float, lon: float) -> str:
    """
    Получить 30-летние климатические нормы (1991-2020).
    Используй как baseline для определения типичного климата.
    """
    client = get_openmeteo_client()
    return await client.get_climate_normals(lat, lon)


VISION_TOOLS = [
    get_climate_history,
    get_seasonal_forecast,
    get_climate_normals
]
