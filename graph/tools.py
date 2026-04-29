from langchain.tools import tool
from rag.retriever import ClimateRetriever 
from pathlib import Path 
from typing import Optional 

_siglip = None 
_climate_retriever = None 
def get_retriever():
    global _climate_retriever
    if _climate_retriever is None:
        _climate_retriever = ClimateRetriever
    return _climate_retriever

@tool 
async def get_city_climate(city_name: str) -> str:
    """Get climate data for a specific city.
    Use this tool when user asks about weather, climate, seasons, or temperature in a city.
    
    Args:
        city_name: Name of the city (e.g., 'Moscow', 'Saint Petersburg')."""
    retriever = get_retriever()
    context = retriever.get_climate_context(city=city_name)
    return context 

@tool 
async def get_climate_by_coordinates(lat:float,lon:float)->str:
    """Get climate data for geographical coordinates.
    Use this when user provides latitude and longitude.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)."""
    retriever = get_retriever()
    context = retriever.get_climate_context(lat=lat,lon=lon)
    return context 

@tool 
async def find_similar_climate(city_name: str) ->str:
    """Find cities with similar climate patterns.
    Use this when user wants to compare climates or find places with similar weather.
    
    Args:
        city_name: Name of the reference city."""
    retriever = get_retriever()
    similar = retriever.find_similar_cities(city_name,top_k=3)
    if not similar:
        return f"No similar cities for {city_name}"
    result = f"**Cities with climate similar to {city_name}:**\n\n"
    for city, data, score in similar:
        monthly = data.get('monthly', {})
        jan_temp = monthly.get('January', {}).get('temp', 'N/A')
        jul_temp = monthly.get('July', {}).get('temp', 'N/A')
        result += f"• **{city}** (match: {score:.2f})\n"
        result += f"  → Winter: {jan_temp}°C, Summer: {jul_temp}°C\n"
    return result

@tool
async def search_climate_by_description(description: str, top_k: int = 3) -> str:
    """
    Search for cities by climate description (semantic search).
    Use this when user describes desired climate in natural language.
    
    Examples of descriptions:
    - 'warm with snowy winters'
    - 'mild climate without extreme temperatures'
    - 'Mediterranean climate'
    - 'cold and dry'
    
    Args:
        description: Natural language description of desired climate
        top_k: Number of results to return (default: 3)
    """
    retriever = get_retriever()
    if hasattr(retriever, 'search_by_text'):
        results = retriever.search_by_text(description, top_k=top_k)
        if not results:
            return f"No cities found matching '{description}'"
        response = f"**Cities matching '{description}':**\n\n"
        for city, data, score in results:
            monthly = data.get('monthly', {})
            winter_temps = []
            for month in ["December", "January", "February"]:
                if month in monthly:
                    winter_temps.append(monthly[month].get('temp', 0))
            avg_winter = sum(winter_temps) / len(winter_temps) if winter_temps else 0
            response += f"• **{city}** (relevance: {score:.2f})\n"
            response += f"  → Average winter: {avg_winter:.1f}°C\n"
        return response
    else:
        return "Semantic search not available"

@tool
async def compare_cities_climate(city1: str, city2: str) -> str:
    """
    Compare climate between two cities.
    Use this when user wants to compare weather patterns, decide where to travel, etc.
    
    Args:
        city1: First city name
        city2: Second city name
    """
    retriever = get_retriever()
    data1 = retriever.find_city_name(city1)
    data2 = retriever.find_city_name(city2)
    if not data1:
        return f"City '{city1}' not found"
    if not data2:
        return f"City '{city2}' not found"
    city1_name = data1.get('city', city1)
    city2_name = data2.get('city', city2)
    monthly1 = data1.get('monthly', {})
    monthly2 = data2.get('monthly', {})
    seasons = {
        "December-February": ["December", "January", "February"],
        "March-May": ["March", "April", "May"],
        "June-August": ["June", "July", "August"],
        "September-November": ["September", "October", "November"]
    }
    result = f" **Climate Comparison: {city1_name} vs {city2_name}**\n\n"
    for season_name, months in seasons.items():
        temps1 = [monthly1.get(m, {}).get('temp', 0) for m in months if m in monthly1]
        temps2 = [monthly2.get(m, {}).get('temp', 0) for m in months if m in monthly2]
        if temps1 and temps2:
            avg1 = sum(temps1) / len(temps1)
            avg2 = sum(temps2) / len(temps2)
            diff = avg1 - avg2
            result += f"**{season_name}:** {avg1:.1f}°C vs {avg2:.1f}°C "
            if diff > 0:
                result += f"({city1_name} warmer by {diff:.1f}°C)\n"
            elif diff < 0:
                result += f"({city2_name} warmer by {-diff:.1f}°C)\n"
            else:
                result += f"(equal)\n"
    return result
    
ALL_TOOLS = [
    compare_cities_climate,
    search_climate_by_description,
    find_similar_climate,
    get_climate_by_coordinates,
    get_city_climate
]