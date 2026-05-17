
import openai
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio
import os 
import base64
import json 

from graph.tools import (
    get_climate_history, 
    get_seasonal_forecast, 
    get_climate_normals, 
    find_similar_cities
)

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
            "description": "Get historical climate data for specific coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "year": {"type": "integer", "default": 2023}
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_forecast",
            "description": "Get seasonal forecast for next 7 months.",
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
            "description": "Get 30-year climate normals.",
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
            "description": "Find cities with similar climate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "top_k": {"type": "integer", "default": 3}
                }
            }
        }
    }
]
TOOLS_MAP = {
    "get_climate_history": get_climate_history,
    "get_seasonal_forecast": get_seasonal_forecast,
    "get_climate_normals": get_climate_normals,
    "find_similar_cities": find_similar_cities,
}

async def call_tool(tool_name: str, **kwargs):
    logger.info(f" Calling tool: {tool_name} with args: {kwargs}")
    
    tool = TOOLS_MAP.get(tool_name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    try:
        if hasattr(tool, 'ainvoke'):
            result = await tool.ainvoke(kwargs)
        else:
            result = await tool(**kwargs)
        return result
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return json.dumps({"error": str(e)})

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
    """
    Анализирует фото с возможностью вызова инструментов
    """
    
    has_coordinates = lat is not None and lon is not None
    
    print("  Computing image hash...", flush=True)
    hash_val = image_hash(path)
    cache_key = f'vision_tools:{hash_val}:{lat}:{lon}:{city}:{temperature}'
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
    
    location_txt = location(lat, lon, city, temperature) if has_coordinates else ""
    siglip_info = ""
    if siglip_prediction:
        siglip_season = siglip_prediction.get('season', 'unknown')
        siglip_conf = siglip_prediction.get('confidence', 0)
        siglip_info = f"""
**🔬 SigLIP PREDICTION (PRIORITY):**
- Season: {siglip_season}
- Confidence: {siglip_conf:.1%}
- Use this as primary reference!
"""
    
    system_prompt = """You are a vision AI that analyzes images to determine the season.

**IMPORTANT RULES:**
1. If you need additional climate data to make an accurate decision, use the available tools.
2. You can call multiple tools before giving your final answer.
3. After receiving tool results, analyze them and respond with JSON.
4. Do NOT include any text outside the JSON in your final response.
5. Final response format: {"season": "winter/spring/summer/autumn", "month": "month name", "confidence": "high/medium/low"}

Available tools:
- get_climate_history: Get historical climate data for coordinates
- get_seasonal_forecast: Get expected seasonal conditions
- get_climate_normals: Get 30-year baseline averages
- find_similar_cities: Find cities with similar climate
"""
    
    user_prompt = f"""
{location_txt}
{siglip_info}

Analyze this image and determine the season and month.

If you need climate data for this location, call the appropriate tool first.
Then provide your final answer as JSON.

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Final answer format: {{"season": "season", "month": "month", "confidence": "high"}}

What season is shown in this image?"""
    
    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
        response = await client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            tools=vision_tools if has_coordinates else None,
            tool_choice="auto" if has_coordinates else "none",
            max_tokens=512,
        )
        
        response_message = response.choices[0].message
        while response_message.tool_calls:
            logger.info(f" Model requested {len(response_message.tool_calls)} tool calls")
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                tool_result = await call_tool(tool_call.function.name, **args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            response = await client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=messages,
                tools=vision_tools if has_coordinates else None,
                tool_choice="auto" if has_coordinates else "none",
                max_tokens=512,
            )
            response_message = response.choices[0].message
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