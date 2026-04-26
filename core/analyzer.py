import ollama 
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio

async def analyze_photo(
        cfg: DictConfig,
        path: str,
        lat:float=None,
        lon:float=None,
        city: str = None,
        ollama_client: ollama.Client = None,
        climate_context: str = ""
)->str:
    """Анализирует фото с кэшем"""
    if ollama_client is None:
        ollama_client = ollama.Client(host=cfg.ollama.host)
    hash = image_hash(path)
    cache_key = f'ollama:{hash}:{lat}:{lon}:{city}:{hash(climate_context)}'
    result = ollama_cache.get(cache_key)
    if result:
        logger.info("Ollama response from cache")
        metrics.track_cache_hit()
        return result
    metrics.track_cache_miss()
    logger.info(f"Calling Ollama")
    metrics.track_api_calls("ollama")
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
    try:
      response = await ollama_client.chat(
        model=cfg.model.name,
        messages=[{
            'role':'user',
            'content':prompt,
            'images':[path]
        }]
      )

      result = response['message']['content']
      ollama_cache.set(cache_key,result,ttl=cfg.model.get('cache_ttl', 3600))
      return result
    except Exception as e:
        logger.error(f"Ollama err:{e}")
        metrics.track_error("ollama_error")
        return ''

    