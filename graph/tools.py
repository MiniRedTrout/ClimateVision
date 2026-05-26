import json

from langchain_core.tools import tool

from utils.logger import logger

try:
    from core.mcp_client import OpenMeteoMCPClient
except ImportError:
    OpenMeteoMCPClient = None

from rag.vector_store import VectorStore

_openmeteo_client = None


def get_openmeteo_client():
    global _openmeteo_client
    if _openmeteo_client is None:
        _openmeteo_client = OpenMeteoMCPClient()
    return _openmeteo_client


@tool
async def find_similar_cities(
    lat: float = None, lon: float = None, top_k: int = 3
) -> str:
    """
    Находит города с похожим климатом.
    Используй когда хочешь узнать похожие по климату города
    Args:
        lat: Широта (если нет города)
        lon: Долгота (если нет города)
        top_k: Количество результатов (по умолчанию 3)

    Returns:
        JSON строка со списком похожих городов
    """
    try:
        vector_store = VectorStore()
        if lat is not None and lon is not None:
            similar = vector_store.find_similar_by_climate(lat, lon, top_k=top_k)
            search_type = f"координатам ({lat}, {lon})"
        else:
            return json.dumps(
                {"status": "error", "message": "Укажите city или (lat, lon)"},
                ensure_ascii=False,
            )

        if not similar:
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Не найдено городов с климатом, похожим на {search_type}",
                    "cities": [],
                },
                ensure_ascii=False,
            )

        result = {"status": "success", "search_type": search_type, "cities": []}

        for city_key, city_data, score in similar:
            result["cities"].append(
                {
                    "name": city_data.get("city", city_key),
                    "country": city_data.get("country", ""),
                    "similarity": round(score * 100, 1),
                }
            )

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@tool
async def get_climate_history(lat: float, lon: float, year: int = 2023) -> str:
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
    get_climate_normals,
    find_similar_cities,
]
