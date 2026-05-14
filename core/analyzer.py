import openai
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio
import os 
import base64
import json 
from tools import get_climate_history, get_seasonal_forecast, get_climate_normals

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
                    "lat": {
                        "type": "number", 
                        "description": "Latitude coordinate"
                    },
                    "lon": {
                        "type": "number", 
                        "description": "Longitude coordinate"
                    },
                    "year": {
                        "type": "integer", 
                        "description": "Year for climate data (default: 2023)",
                        "default": 2023
                    }
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
                    "lat": {
                        "type": "number", 
                        "description": "Latitude coordinate"
                    },
                    "lon": {
                        "type": "number", 
                        "description": "Longitude coordinate"
                    }
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
                    "lat": {
                        "type": "number", 
                        "description": "Latitude coordinate"
                    },
                    "lon": {
                        "type": "number", 
                        "description": "Longitude coordinate"
                    }
                },
                "required": ["lat", "lon"]
            }
        }
    }
]

async def analyze_photo(
        cfg: DictConfig,
        path: str,
        lat:float=None,
        lon:float=None,
        city: str = None,
        temperature: str = None,
        climate_context: str = ""
)->str:
    """Анализирует фото с кэшем"""
    print("  Computing image hash...", flush=True)
    hash_val = image_hash(path)
    cache_key = f'ollama:{hash_val}:{lat}:{lon}:{city}:{temperature}{hash(climate_context)}'
    print("  Cache key created", flush=True)
    result = ollama_cache.get(cache_key)
    if result:
        logger.info("Ollama response from cache")
        metrics.track_cache_hit()
        return result
    metrics.track_cache_miss()
    logger.info(f"Calling Ollama")
    metrics.track_api_call("ollama")
    location_txt = location(lat,lon,city,temperature)
    if climate_context:
        climate_section = f"""
CLIMATE REFERENCE (use as additional info, not as strict rule):
{climate_context}

Compare what you see in the image with this climate reference.
If there's a contradiction, trust the VISUAL evidence from the image more.
"""
    else:
        climate_section = ''
    prompt = f"""
{location_txt}

Analyze this image and determine the season and month.

**IMPORTANT**: You have access to climate tools to help you make better determination:
- `get_climate_normals` - Use this FIRST to understand typical climate for this location
- `get_seasonal_forecast` - Use this to see expected conditions for upcoming months
- `get_climate_history` - Use this if you need data from a specific year

Strategy for best results:
1. First call `get_climate_normals` to understand baseline temperatures and precipitation by month
2. If the image shows unusual conditions (e.g., flowers in winter), you might want to check seasonal forecast
3. Use the climate data to confirm or adjust your visual assessment

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

If you cannot determine, use "unknown" for season and "unknown" for month.

Your response:"""
    
    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
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
            tools=vision_tools,
            tool_choice="auto",
            max_tokens=512,
        )
        response_message = response.choices[0].message
        if response_message.tool_calls:
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
                
                if tool_call.function.name == "get_climate_history":
                    tool_result = await get_climate_history(**args)
                elif tool_call.function.name == "get_seasonal_forecast":
                    tool_result = await get_seasonal_forecast(**args)
                elif tool_call.function.name == "get_climate_normals":
                    tool_result = await get_climate_normals(**args)
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
            parsed = json.loads(result)
            if "season" in parsed and "month" in parsed:
                ollama_cache.set(cache_key, result, ttl=cfg.model.get('cache_ttl', 3600))
                return result
            else:
                logger.warning(f"Invalid response format: {result}")
                return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})
        except:
            logger.warning(f"Failed to parse response: {result}")
            return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})
            
    except Exception as e:
        logger.error(f"Vision model error: {e}")
        metrics.track_error("vision_model_error")
        return json.dumps({"season": "unknown", "month": "unknown", "confidence": "low"})
    