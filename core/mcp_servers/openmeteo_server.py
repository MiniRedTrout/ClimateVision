import asyncio
import httpx
from datetime import datetime, timedelta
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types
from collections import defaultdict
import sys

server = Server("openmeteo-server")

# Добавляем вывод в stderr для отладки
print("OpenMeteo MCP Server starting...", file=sys.stderr)

@server.list_tools()
async def list_tools():
    print("list_tools called", file=sys.stderr)
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
            description="Get seasonal forecast for next 7 months (SEAS5 model). Best for understanding expected seasonal conditions and anomalies",
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
            name="get_climate_normals",
            description="Get 30-year climate normals (1991-2020). Provides baseline average temperatures and precipitation by month",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"}
                },
                "required": ["lat", "lon"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    print(f"call_tool: {name}, arguments: {arguments}", file=sys.stderr)
    
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
            text = f"""**Current Weather**
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
            
            text = f"**Climate Data for {year}**\n\n"
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
            text = f"**{days}-Day Forecast**\n\n"
            for i in range(len(data["daily"]["time"])):
                date = data["daily"]["time"][i]
                temp_max = data["daily"]["temperature_2m_max"][i]
                temp_min = data["daily"]["temperature_2m_min"][i]
                precip = data["daily"]["precipitation_sum"][i]
                text += f"**{date}**: {temp_min:.0f}°C / {temp_max:.0f}°C, Precipitation: {precip}mm\n"
            
            return {"content": [{"type": "text", "text": text}]}
    
    elif name == "get_seasonal_forecast":
        lat = arguments["lat"]
        lon = arguments["lon"]
        
        async with httpx.AsyncClient() as client:
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
            
            # ФIX: исправлен парсинг даты и убран return внутри цикла
            for i in range(len(monthly.get("time", []))):
                date_str = monthly["time"][i]
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")  # FIXED: %Y вместо %X
                    month_name = month_names[date_obj.month - 1]
                    year = date_obj.year 
                except Exception as e:
                    print(f"Error parsing date {date_str}: {e}", file=sys.stderr)
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
            
            text += f"\n*Note: SEAS5 provides probabilistic forecasts. Values shown are ensemble means.*"
            return {"content": [{"type": "text", "text": text}]}
        else:
            return {"content": [{"type": "text", "text": f"Seasonal forecast data not available for {lat:.2f}, {lon:.2f}"}]}
    
    elif name == "get_climate_normals":
        lat = arguments["lat"]
        lon = arguments["lon"]
        start_date = "1991-01-01"
        end_date = "2020-12-31"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": ["temperature_2m_max", "temperature_2m_min", 
                             "temperature_2m_mean", "precipitation_sum", "snowfall_sum"],
                    "timezone": "auto"
                }
            )
            data = response.json()
        
        if "daily" in data:
            monthly_stats = defaultdict(lambda: {
                "temps_mean": [],
                "temps_max": [],
                "temps_min": [],
                "precip": [],
                "snow": []
            })
            
            daily = data["daily"]
            for i, date_str in enumerate(daily.get("time", [])):
                try:
                    month = datetime.strptime(date_str, "%Y-%m-%d").month
                    
                    temp_mean = daily.get("temperature_2m_mean", [None])[i]
                    temp_max = daily.get("temperature_2m_max", [None])[i]
                    temp_min = daily.get("temperature_2m_min", [None])[i]
                    precip = daily.get("precipitation_sum", [None])[i]
                    snow = daily.get("snowfall_sum", [None])[i]
                    
                    if temp_mean is not None:
                        monthly_stats[month]["temps_mean"].append(temp_mean)
                    if temp_max is not None:
                        monthly_stats[month]["temps_max"].append(temp_max)
                    if temp_min is not None:
                        monthly_stats[month]["temps_min"].append(temp_min)
                    if precip is not None:
                        monthly_stats[month]["precip"].append(precip)
                    if snow is not None:
                        monthly_stats[month]["snow"].append(snow)
                        
                except Exception as e:
                    continue
            
            month_names = {1: "January", 2: "February", 3: "March", 4: "April", 
                          5: "May", 6: "June", 7: "July", 8: "August",
                          9: "September", 10: "October", 11: "November", 12: "December"}
            
            text = f"**30-Year Climate Normals (1991-2020)**\n"
            text += f"Location: {lat:.2f}, {lon:.2f}\n"
            text += f"Period: 30 years (1991-2020)\n"
            text += f"Source: ERA5 reanalysis\n\n"
            
            for month in range(1, 13):
                stats = monthly_stats.get(month, {})
                
                if stats.get("temps_mean"):
                    avg_mean = sum(stats["temps_mean"]) / len(stats["temps_mean"])
                    avg_max = sum(stats["temps_max"]) / len(stats["temps_max"]) if stats["temps_max"] else None
                    avg_min = sum(stats["temps_min"]) / len(stats["temps_min"]) if stats["temps_min"] else None
                    total_precip = sum(stats["precip"]) if stats["precip"] else 0
                    total_snow = sum(stats["snow"]) if stats["snow"] else 0
                    
                    text += f"**{month_names[month]}**:\n"
                    text += f"  Avg Mean Temp: {avg_mean:.1f}°C\n"
                    if avg_max:
                        text += f"  Avg Max Temp: {avg_max:.1f}°C\n"
                    if avg_min:
                        text += f"  Avg Min Temp: {avg_min:.1f}°C\n"
                    text += f"  Total Precipitation: {total_precip:.0f}mm\n"
                    if total_snow > 0:
                        text += f"  Total Snowfall: {total_snow:.0f}mm\n"
                    text += "\n"
            
            text += f"*These are baseline averages. Compare with current/historical data to identify anomalies.*"
            return {"content": [{"type": "text", "text": text}]}
        else:
            return {"content": [{"type": "text", "text": f"Climate normals not available for {lat:.2f}, {lon:.2f}"}]}
    
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

if __name__ == "__main__":
    asyncio.run(server.run())