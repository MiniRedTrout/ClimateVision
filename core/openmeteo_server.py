
import asyncio
import httpx
from datetime import datetime, timedelta
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types

server = Server("openmeteo-server")

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="get_current_weather",
            description="Get current weather for any location on Earth",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"}
                },
                "required": ["lat", "lon"]
            }
        ),
        types.Tool(
            name="get_climate_history",
            description="Get historical climate data (temperature, snow, rain) for past years",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "year": {"type": "integer", "description": "Year (2020-2024)", "default": 2023}
                },
                "required": ["lat", "lon"]
            }
        ),
        types.Tool(
            name="get_forecast",
            description="Get weather forecast for next 7 days",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "days": {"type": "integer", "default": 3}
                },
                "required": ["lat", "lon"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    
    if name == "get_current_weather":
        lat = arguments["lat"]
        lon = arguments["lon"]
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True
                }
            )
            data = response.json()
        
        if "current_weather" in data:
            w = data["current_weather"]
            text = f""" **Current Weather**
 Coordinates: {lat:.2f}, {lon:.2f}
 Temperature: {w.get('temperature', 'N/A')}°C
 Wind Speed: {w.get('windspeed', 'N/A')} km/h
 Wind Direction: {w.get('winddirection', 'N/A')}°
 Time: {w.get('time', 'N/A')}"""
            return {"content": [{"type": "text", "text": text}]}
    
    elif name == "get_climate_history":
        lat = arguments["lat"]
        lon = arguments["lon"]
        year = arguments.get("year", 2023)
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": ["temperature_2m_mean", "snowfall_sum", "rain_sum"],
                    "timezone": "auto"
                }
            )
            data = response.json()
        
        if "daily" in data:
            months_data = {}
            for i, date_str in enumerate(data["daily"]["time"]):
                month = datetime.strptime(date_str, "%Y-%m-%d").month
                temp = data["daily"]["temperature_2m_mean"][i]
                snow = data["daily"]["snowfall_sum"][i]
                rain = data["daily"]["rain_sum"][i]
                
                if month not in months_data:
                    months_data[month] = {"temps": [], "snow": 0, "rain": 0}
                months_data[month]["temps"].append(temp)
                months_data[month]["snow"] += snow or 0
                months_data[month]["rain"] += rain or 0
            
            text = f" **Climate Data for {year}**\n\n"
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            
            for month in [12, 1, 2, 3, 4]: 
                if month in months_data:
                    avg_temp = sum(months_data[month]["temps"]) / len(months_data[month]["temps"])
                    text += f"**{month_names[month]}**: {avg_temp:.1f}°C, Snow: {months_data[month]['snow']:.0f}mm, Rain: {months_data[month]['rain']:.0f}mm\n"
            
            return {"content": [{"type": "text", "text": text}]}
    
    elif name == "get_forecast":
        lat = arguments["lat"]
        lon = arguments["lon"]
        days = arguments.get("days", 3)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                    "forecast_days": days,
                    "timezone": "auto"
                }
            )
            data = response.json()
        
        if "daily" in data:
            text = f" **{days}-Day Forecast**\n\n"
            for i in range(len(data["daily"]["time"])):
                date = data["daily"]["time"][i]
                temp_max = data["daily"]["temperature_2m_max"][i]
                temp_min = data["daily"]["temperature_2m_min"][i]
                precip = data["daily"]["precipitation_sum"][i]
                text += f"**{date}**: {temp_min:.0f}°C / {temp_max:.0f}°C, Precipitation: {precip}mm\n"
            
            return {"content": [{"type": "text", "text": text}]}
    
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

if __name__ == "__main__":
    asyncio.run(server.run())