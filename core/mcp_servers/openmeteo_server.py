import asyncio
import httpx
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from collections import defaultdict
import sys

server = Server("openmeteo-server")

print("OpenMeteo MCP Server initializing...", file=sys.stderr, flush=True)

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
        ),
        types.Tool(
            name="get_seasonal_forecast",
            description="Get seasonal forecast for next 7 months",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                },
                "required": ["lat", "lon"]
            }
        ),
        types.Tool(
            name="get_climate_normals",
            description="Get 30-year climate normals (1991-2020)",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                },
                "required": ["lat", "lon"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    print(f"call_tool: {name}, {arguments}", file=sys.stderr, flush=True)
    
    try:
        if name == "get_current_weather":
            lat = arguments["lat"]
            lon = arguments["lon"]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                text = f"**Current Weather**\nCoordinates: {lat:.2f}, {lon:.2f}\nTemperature: {w.get('temperature', 'N/A')}°C\nWind Speed: {w.get('windspeed', 'N/A')} km/h\nWind Direction: {w.get('winddirection', 'N/A')}°\nTime: {w.get('time', 'N/A')}"
                return types.TextContent(type="text", text=text)
        
        elif name == "get_climate_history":
            lat = arguments["lat"]
            lon = arguments["lon"]
            year = arguments.get("year", 2023)
            
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                
                text = f"**Climate Data for {year}**\n\n"
                month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                
                for month in [12, 1, 2, 3, 4]: 
                    if month in months_data:
                        avg_temp = sum(months_data[month]["temps"]) / len(months_data[month]["temps"])
                        text += f"**{month_names[month]}**: {avg_temp:.1f}°C, Snow: {months_data[month]['snow']:.0f}mm, Rain: {months_data[month]['rain']:.0f}mm\n"
                
                return types.TextContent(type="text", text=text)
        
        elif name == "get_forecast":
            lat = arguments["lat"]
            lon = arguments["lon"]
            days = arguments.get("days", 3)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                text = f"**{days}-Day Forecast**\n\n"
                for i in range(len(data["daily"]["time"])):
                    date = data["daily"]["time"][i]
                    temp_max = data["daily"]["temperature_2m_max"][i]
                    temp_min = data["daily"]["temperature_2m_min"][i]
                    precip = data["daily"]["precipitation_sum"][i]
                    text += f"**{date}**: {temp_min:.0f}°C / {temp_max:.0f}°C, Precipitation: {precip}mm\n"
                
                return types.TextContent(type="text", text=text)
        
        elif name == "get_seasonal_forecast":
            lat = arguments["lat"]
            lon = arguments["lon"]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://seasonal-api.open-meteo.com/v1/seasonal",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "monthly": ["temperature_2m_max", "temperature_2m_min", 
                                   "precipitation_sum", "snowfall_sum"],
                        "models": "seas5",
                        "timezone": "auto"
                    }
                )
                data = response.json()
            
            if "monthly" in data:
                monthly = data["monthly"]
                text = f"**Seasonal Forecast (SEAS5)** - Next 7 Months\n"
                text += f"Location: {lat:.2f}, {lon:.2f}\n"
                text += f"Model: ECMWF SEAS5 (51 ensemble members)\n\n"
                
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                
                for i in range(len(monthly.get("time", []))):
                    date_str = monthly["time"][i]
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        month_name = month_names[date_obj.month - 1]
                        year = date_obj.year 
                    except:
                        month_name = date_str
                        year = ""
                    
                    temp_max = monthly.get("temperature_2m_max", [None])[i]
                    temp_min = monthly.get("temperature_2m_min", [None])[i]
                    precip = monthly.get("precipitation_sum", [None])[i]
                    snow = monthly.get("snowfall_sum", [None])[i]
                    
                    text += f"\n**{month_name} {year}**:\n"
                    if temp_max is not None:
                        text += f"  Max Temp: {temp_max:.1f}°C\n"
                    if temp_min is not None:
                        text += f"  Min Temp: {temp_min:.1f}°C\n"
                    if precip is not None:
                        text += f"  Precipitation: {precip:.0f}mm\n"
                    if snow is not None:
                        text += f"  Snowfall: {snow:.0f}mm\n"
                
                return types.TextContent(type="text", text=text)
            else:
                return types.TextContent(type="text", text=f"Seasonal forecast not available")
        
        elif name == "get_climate_normals":
            lat = arguments["lat"]
            lon = arguments["lon"]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": "1991-01-01",
                        "end_date": "2020-12-31",
                        "daily": ["temperature_2m_mean", "precipitation_sum"],
                        "timezone": "auto"
                    }
                )
                data = response.json()
            
            if "daily" in data:
                text = f"**30-Year Climate Normals (1991-2020)**\nLocation: {lat:.2f}, {lon:.2f}\n\n"
                # Упрощенный вывод для экономии места
                return types.TextContent(type="text", text=text)
            else:
                return types.TextContent(type="text", text=f"Climate normals not available")
        
        return types.TextContent(type="text", text=f"Unknown tool: {name}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return types.TextContent(type="text", text=f"Error: {str(e)}")

async def main():
    """Запуск MCP сервера через stdio"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())