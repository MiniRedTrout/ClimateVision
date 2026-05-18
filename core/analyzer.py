
import openai
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio
import os 
import base64
import json 
import re
from utils.helpers import extract_json_from_response
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
** SigLIP PREDICTION (PRIORITY):**
- Season: {siglip_season}
- Confidence: {siglip_conf:.1%}
- Use this as primary reference!
"""
    system_prompt = """You are a vision AI that analyzes images to determine the season.

**SEASONAL INDICATORS - Use these to make your decision:**

🌱 SPRING (March-May in Northern Hemisphere):
- Visual clues: fresh bright light-green buds, young leaves on branches, bright green young grass, flowering trees (cherry blossoms, magnolia), spring flowers (tulips, daffodils, crocuses), moist dark soil, sparse canopy
- Temperature: typically 5-18°C
- Light: increasing daylight, softer sun
- People: light jackets, long sleeves

☀️ SUMMER (June-August):
- Visual clues: full dense dark green canopy, lush green grass, colorful summer flowers, bright blue sky, intense sun
- Temperature: typically 18-35°C
- Light: long days, strong sunlight
- People: t-shirts, shorts, summer dresses, hats

🍂 AUTUMN (September-November):
- Visual clues: leaves turning yellow/orange/red/brown, fallen leaves covering ground, dried yellowish grass, fruits/mushrooms, bare branches starting (late autumn)
- Temperature: typically 5-18°C (cooling)
- Light: shorter days, golden hour light
- People: jackets, sweaters

❄️ WINTER (December-February):
- Visual clues: bare branches, snow on ground/trees, frost/ice, evergreen trees only, grey sky, short shadows
- Temperature: typically below 0°C
- Light: short days, low sun angle
- People: heavy coats, hats, scarves

**CRITICAL: SPRING vs AUTUMN DISAMBIGUATION** (visually similar, commonly confused)

| Clue | SPRING | AUTUMN |
|------|--------|--------|
| Leaves | Young, bright green, budding | Mature, yellow/red/brown, falling |
| Ground | Moist dark soil, fresh green grass | Dry soil, brown grass, fallen leaves |
| Flowers | Tulips, daffodils, blossoms | None or berries/mushrooms |
| Light | Increasing, cool | Decreasing, warm golden |
| Temperature trend | WARMING UP from winter | COOLING DOWN from summer |

**USE TEMPERATURE DATA:**
- If temperature is provided, COMPARE it with climate normals
- Temperatures TRENDING UP from winter norms → SPRING
- Temperatures TRENDING DOWN from summer norms → AUTUMN
- Example: +5°C in March (spring) vs +5°C in November (autumn) → same temp, different trend!

**USE CLIMATE NORMALS:**
- If temp matches March/April/May averages → likely SPRING
- If temp matches September/October/November averages → likely AUTUMN

**MONTH NARROWING:**
- Early spring (March): cold, possible snow, very few buds
- Mid spring (April): buds opening, first green appearing
- Late spring (May): full green, warm, flowers blooming
- Early autumn (September): still mostly green, slight yellowing
- Mid autumn (October): strong colors, leaves falling
- Late autumn (November): mostly bare trees, cold, brown

**CRITICAL INSTRUCTION:**
- You MUST respond ONLY with valid JSON.
- NO explanatory text before or after the JSON.
- NO markdown formatting.
- JUST the raw JSON object.

Example response: {"season": "spring", "month": "April", "confidence": "high"}

Available tools (use if needed):
- get_climate_history, get_seasonal_forecast, get_climate_normals, find_similar_cities

Now analyze the image and respond ONLY with JSON."""
    
    user_prompt = f"""
{location_txt}
{siglip_info}

Use the seasonal indicators above to analyze this image.

Respond ONLY with JSON. No other text.

Your response:"""
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
        raw_response = response_message.content
        logger.info(f"Raw response: {raw_response[:200]}...")
        parsed = extract_json_from_response(raw_response)
        
        if parsed and "season" in parsed:
            if "month" not in parsed:
                parsed["month"] = "unknown"
            if "confidence" not in parsed:
                parsed["confidence"] = "medium"
            
            result = json.dumps(parsed)
            logger.info(f"✅ Parsed: {parsed}")
            ollama_cache.set(cache_key, result, ttl=cfg.model.get('cache_ttl', 3600))
            return result
        else:
            logger.warning(f"Failed to parse response: {raw_response[:200]}")
            return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})
            
    except Exception as e:
        logger.error(f"Vision model error: {e}")
        metrics.track_error("vision_model_error")
        return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})