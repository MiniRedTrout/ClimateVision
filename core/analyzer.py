import openai
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio
import os 
import base64
import json 
from graph.tools import get_climate_history, get_seasonal_forecast, get_climate_normals

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
            "description": "Get historical climate data for specific coordinates for a specific year. Use when you need to know actual weather patterns from a particular year.",
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
            "description": "Get seasonal forecast for next 7 months (SEAS5 model). Use when you need to understand expected seasonal conditions or anomalies for the upcoming months.",
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
            "description": "Get 30-year climate normals (1991-2020). Use as baseline to understand what is typical for this location - average temperatures and precipitation by month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                },
                "required": ["lat", "lon"]
            }
        }
    }
]

async def call_tool(tool_func, **kwargs):
    try:
        if hasattr(tool_func, 'ainvoke'):
            result = await tool_func.ainvoke(kwargs)
        else:
            result = await tool_func(**kwargs)
        return result
    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return json.dumps({"status": "error", "message": str(e)})

async def analyze_photo(
        cfg: DictConfig,
        path: str,
        lat: float = None,
        lon: float = None,
        city: str = None,
        temperature: str = None,
        climate_context: str = ""
) -> str:
    has_coordinates = lat is not None and lon is not None
    
    print("  Computing image hash...", flush=True)
    hash_val = image_hash(path)
    cache_key = f'vision_with_tools:{hash_val}:{lat}:{lon}:{city}:{temperature}:{hash(climate_context)}'
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

Compare what you see in the image with this climate reference.
If there's a contradiction, trust the VISUAL evidence from the image more.
"""
    else:
        climate_section = ""
    if has_coordinates:
        prompt = f"""
{location_txt}
{climate_section}

Analyze this image and determine the season and month.

You have access to climate tools. Coordinates are available: lat={lat}, lon={lon}

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

Your response:"""
    else:
        prompt = f"""
{climate_section}

Analyze this image and determine the season and month.

NO COORDINATES AVAILABLE - You CANNOT use climate tools.

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
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
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
            logger.info(f"Vision model requested {len(response_message.tool_calls)} tool calls")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
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
                logger.info(f"Executing tool: {tool_call.function.name}")
                args = json.loads(tool_call.function.arguments)
                if 'lat' not in args and lat is not None:
                    args['lat'] = lat
                if 'lon' not in args and lon is not None:
                    args['lon'] = lon
                if tool_call.function.name == "get_climate_history":
                    tool_result = await call_tool(get_climate_history, **args)
                elif tool_call.function.name == "get_seasonal_forecast":
                    tool_result = await call_tool(get_seasonal_forecast, **args)
                elif tool_call.function.name == "get_climate_normals":
                    tool_result = await call_tool(get_climate_normals, **args)
                else:
                    tool_result = json.dumps({"error": f"Unknown tool: {tool_call.function.name}"})
                
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