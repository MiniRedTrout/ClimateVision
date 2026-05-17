# core/analyzer.py
import openai
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio
import os 
import base64
import json 
from graph.tools import get_climate_history, get_seasonal_forecast, get_climate_normals, find_similar_cities

GROQ_API_KEY = os.getenv("API_KEY")
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" 
GROQ_API_BASE = "https://api.groq.com/openai/v1"
client = openai.AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_BASE,
)

vision_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_climate_history",
            "description": "Get historical climate data for specific coordinates. Use when you need past climate patterns for this location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"},
                    "year": {"type": "integer", "description": "Year", "default": 2023}
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_forecast",
            "description": "Get seasonal forecast for next 7 months. Use to understand expected seasonal conditions and anomalies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_climate_normals",
            "description": "Get 30-year climate normals (1991-2020). Use as baseline to understand typical climate for this location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_cities",
            "description": "Find cities with similar climate. Use when you need to compare or find analogues for this location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name (e.g., 'Moscow', 'Berlin')"},
                    "lat": {"type": "number", "description": "Latitude (if city not provided)"},
                    "lon": {"type": "number", "description": "Longitude (if city not provided)"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 3}
                }
            }
        }
    }
]

async def call_tool(tool_name: str, **kwargs):
    logger.info(f"Vision model calling tool: {tool_name} with args: {kwargs}")
    
    if tool_name == "get_climate_history":
        return await get_climate_history(**kwargs)
    elif tool_name == "get_seasonal_forecast":
        return await get_seasonal_forecast(**kwargs)
    elif tool_name == "get_climate_normals":
        return await get_climate_normals(**kwargs)
    elif tool_name == "find_similar_cities":
        return await find_similar_cities(**kwargs)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

async def analyze_photo(
        cfg: DictConfig,
        path: str,
        lat: float = None,
        lon: float = None,
        city: str = None,
        temperature: float = None,
        climate_context: str = "",
        siglip_prediction: dict = None
) -> str:
    has_coordinates = lat is not None and lon is not None
    
    print("  Computing image hash...", flush=True)
    hash_val = image_hash(path)
    cache_key = f'vision_with_tools:{hash_val}:{lat}:{lon}:{city}:{temperature}:{hash(climate_context)}'
    if siglip_prediction:
        cache_key += f':{siglip_prediction.get("season", "none")}'
    print("  Cache key created", flush=True)
    
    result = ollama_cache.get(cache_key)
    if result:
        logger.info("Vision response from cache")
        metrics.track_cache_hit()
        return result
    
    metrics.track_cache_miss()
    logger.info(f"Calling Vision model with tools")
    metrics.track_api_call("vision_model")
    
    location_txt = location(lat, lon, city, temperature) if has_coordinates else ""
    
    if climate_context and climate_context != "No climate data available for this location.":
        climate_section = f"""
CLIMATE REFERENCE (use as additional context):
{climate_context}
"""
    else:
        climate_section = ""
    
    # SigLIP секция
    siglip_section = ""
    if siglip_prediction and siglip_prediction.get('season'):
        siglip_season = siglip_prediction['season']
        siglip_confidence = siglip_prediction.get('confidence', 0.7)
        probs = siglip_prediction.get('probabilities', {})
        season_ru = {'winter': 'зима', 'spring': 'весна', 'summer': 'лето', 'autumn': 'осень'}.get(siglip_season, siglip_season)
        
        siglip_section = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   **ВАЖНО: SigLIP АНАЛИЗ (ПРИОРИТЕТ)**                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  SigLIP предсказание: {siglip_season.upper()} ({season_ru})                              ║
║  Уверенность: {siglip_confidence:.1%}                                                  ║
║                                                                               ║
║  Вероятности:                                                                 ║
║  • Зима (winter):   {probs.get('winter', 0):.0%}                                         ║
║  • Весна (spring):  {probs.get('spring', 0):.0%}                                         ║
║  • Лето (summer):   {probs.get('summer', 0):.0%}                                         ║
║  • Осень (autumn):  {probs.get('autumn', 0):.0%}                                         ║
║                                                                               ║
║  SigLIP обучен на реальных климатических данных (температура + координаты)    ║
║  Используй это предсказание как источник!                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
        logger.info(f"📊 SigLIP предсказание: {siglip_season} (уверенность: {siglip_confidence:.1%})")
    if has_coordinates:
        prompt = f"""
{location_txt}
{climate_section}
{siglip_section}

Analyze this image and determine the season and month.

**AVAILABLE TOOLS:**
- get_climate_history: Get past climate data for this location
- get_seasonal_forecast: Get expected seasonal conditions
- get_climate_normals: Get 30-year baseline averages
- find_similar_cities: Find cities with similar climate

Use these tools if you need additional climate data for better accuracy.

**KEY INSTRUCTION:** The SigLIP prediction above has HIGH PRIORITY because it's fine-tuned on real climate data.

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

Your response:"""
    else:
        prompt = f"""
{climate_section}
{siglip_section}

Analyze this image and determine the season and month.

**NO COORDINATES AVAILABLE** - You CANNOT use climate tools that require lat/lon.

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

Your response:"""
    
    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        kwargs = {
            "model": GROQ_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 512,
        }
        if has_coordinates:
            kwargs["tools"] = vision_tools
            kwargs["tool_choice"] = "auto"
        
        response = await client.chat.completions.create(**kwargs)
        response_message = response.choices[0].message
        if has_coordinates and response_message.tool_calls:
            logger.info(f" Vision model requested {len(response_message.tool_calls)} tool calls")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
            
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ]
            })
            
            for tool_call in response_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                tool_result = await call_tool(tool_call.function.name, **args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            second_response = await client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=messages,
                max_tokens=512,
            )
            result = second_response.choices[0].message.content
        else:
            result = response_message.content
        try:
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            parsed = json.loads(result)
            if "season" in parsed and "month" in parsed:
                if "confidence" not in parsed:
                    parsed["confidence"] = "medium"
                
                if siglip_prediction and siglip_prediction.get('season'):
                    if parsed['season'] == siglip_prediction['season']:
                        logger.info(f" Vision согласен с SigLIP: {parsed['season']}")
                    else:
                        logger.warning(f" Vision НЕ согласен с SigLIP! Vision={parsed['season']}, SigLIP={siglip_prediction['season']}")
                
                result = json.dumps(parsed)
                ollama_cache.set(cache_key, result, ttl=cfg.model.get('cache_ttl', 3600))
                return result
            else:
                logger.warning(f"Invalid response format: {result}")
                return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response: {result}, error: {e}")
            return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})
            
    except Exception as e:
        logger.error(f"Vision model error: {e}")
        metrics.track_error("vision_model_error")
        return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})