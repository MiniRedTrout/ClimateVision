import openai
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio
import os 
import base64

GROQ_API_KEY = os.getenv("API_KEY")
GROQ_MODEL_NAME = "llama-4-scout" 
GROQ_API_BASE = "https://api.groq.com/openai/v1"
client = openai.AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_BASE,
)
async def analyze_photo(
        cfg: DictConfig,
        path: str,
        lat:float=None,
        lon:float=None,
        city: str = None,
        climate_context: str = ""
)->str:
    """Анализирует фото с кэшем"""
    print("  Computing image hash...", flush=True)
    hash_val = image_hash(path)
    cache_key = f'ollama:{hash_val}:{lat}:{lon}:{city}:{hash(climate_context)}'
    print("  Cache key created", flush=True)
    result = ollama_cache.get(cache_key)
    if result:
        logger.info("Ollama response from cache")
        metrics.track_cache_hit()
        return result
    metrics.track_cache_miss()
    logger.info(f"Calling Ollama")
    metrics.track_api_call("ollama")
    location_txt = location(lat,lon,city)
    if climate_context:
        climate_section = f"""
CLIMATE CONTEXT:
{climate_context}

Use this climate information to improve your analysis.
"""
    else:
        climate_section = ''
    prompt = f"""
{location_txt}
{climate_section}

Analyze this image. You MUST determine BOTH season AND month.
If you cannot determine, use "unknown" for season and "unknown" for month.

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

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
            max_tokens=512,
      )

      result = response['message']['content']
      ollama_cache.set(cache_key,result,ttl=cfg.model.get('cache_ttl', 3600))
      return result
    except Exception as e:
        logger.error(f"Ollama err:{e}")
        metrics.track_error("ollama_error")
        return ''

    