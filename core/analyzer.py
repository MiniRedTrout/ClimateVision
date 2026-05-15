import openai
from cache import ollama_cache 
from utils import logger, metrics, parse, image_hash, location
from omegaconf import DictConfig
import asyncio
import os 
import base64
import json 

GROQ_API_KEY = os.getenv("API_KEY")
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" 
GROQ_API_BASE = "https://api.groq.com/openai/v1"
client = openai.AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_BASE,
)

async def analyze_photo(
        cfg: DictConfig,
        path: str,
        lat: float = None,
        lon: float = None,
        city: str = None,
        temperature: float = None,
        climate_context: str = "",
        siglip_prediction: dict = None  # НОВЫЙ ПАРАМЕТР
) -> str:
    has_coordinates = lat is not None and lon is not None
    
    print("  Computing image hash...", flush=True)
    hash_val = image_hash(path)
    cache_key = f'vision_with_siglip:{hash_val}:{lat}:{lon}:{city}:{temperature}:{hash(climate_context)}'
    if siglip_prediction:
        cache_key += f':{siglip_prediction.get("season", "none")}'
    print("  Cache key created", flush=True)
    
    result = ollama_cache.get(cache_key)
    if result:
        logger.info("Vision response from cache")
        metrics.track_cache_hit()
        return result
    
    metrics.track_cache_miss()
    logger.info(f"Calling Vision model with SigLIP guidance")
    metrics.track_api_call("vision_model")
    
    location_txt = location(lat, lon, city, temperature) if has_coordinates else ""
    if climate_context and climate_context != "No climate data available for this location.":
        climate_section = f"""
CLIMATE REFERENCE (use as additional context):
{climate_context}

Compare what you see in the image with this climate reference.
"""
    else:
        climate_section = ""
    siglip_section = ""
    if siglip_prediction and siglip_prediction.get('season'):
        siglip_season = siglip_prediction['season']
        siglip_confidence = siglip_prediction.get('confidence', 0.7)
        probs = siglip_prediction.get('probabilities', {})
        season_ru = {'winter': 'зима', 'spring': 'весна', 'summer': 'лето', 'autumn': 'осень'}.get(siglip_season, siglip_season)
        
        siglip_section = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  🔬 **ВАЖНО: SigLIP АНАЛИЗ**                                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  SigLIP - это модель, дообученная на РЕАЛЬНЫХ КЛИМАТИЧЕСКИХ ДАННЫХ:           ║
║  • Учитывает координаты (широту/долготу)                                      ║
║  • Учитывает температуру воздуха                                              ║
║  • Обучена на многолетних метеоданных                                         ║
║                                                                               ║
║  **ПРЕДСКАЗАНИЕ SigLIP: {siglip_season.upper()} ({season_ru})**               ║
║  **Уверенность SigLIP: {siglip_confidence:.1%}**                              ║
║                                                                               ║
║  Детальные вероятности SigLIP:                                                ║
║  • Зима (winter):   {probs.get('winter', 0):.0%}                              ║
║  • Весна (spring):  {probs.get('spring', 0):.0%}                              ║
║  • Лето (summer):   {probs.get('summer', 0):.0%}                              ║
║  • Осень (autumn):  {probs.get('autumn', 0):.0%}                              ║
║                                                                               ║
║                                                                               ║
║                                                                               ║
║                                                                               ║
║                                                                               ║
║                                                                               ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
        logger.info(f"📊 В Vision модель передано SigLIP предсказание: {siglip_season} (уверенность: {siglip_confidence:.1%})")
    if has_coordinates:
        prompt = f"""
{location_txt}
{climate_section}
{siglip_section}

Analyze this image and determine the season and month.

**KEY INSTRUCTION:**
The SigLIP analysis above has PRIORITY  because it's fine-tuned on real climate data.

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

**KEY INSTRUCTION:**
The SigLIP analysis above has PRIORITY because it's fine-tuned on real climate data.

NO COORDINATES AVAILABLE.

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
        
        result = response.choices[0].message.content
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
                        logger.info(f"Vision СОГЛАСЕН с SigLIP: {parsed['season']}")
                    else:
                        logger.warning(f"Vision НЕ СОГЛАСЕН с SigLIP! Vision={parsed['season']}, SigLIP={siglip_prediction['season']}")
                
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