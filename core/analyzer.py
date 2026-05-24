import base64
import json
import os

from dotenv import load_dotenv

load_dotenv()

import openai
from omegaconf import DictConfig

from cache import ollama_cache
from utils import image_hash, location, logger, metrics

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = openai.AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_API_BASE,
)

SEASON_SPRING_AUTUMN_GUIDE = """
CRITICAL: SPRING vs AUTUMN DISAMBIGUATION

These two seasons are visually similar and commonly confused. Use ALL available clues:

VISUAL CLUES:
- SPRING indicators:
  * Fresh, bright light-green buds and young leaves on branches
  * Bright green young grass appearing through dark soil
  * Flowering trees (cherry blossoms, magnolia, lilac)
  * Spring flowers (tulips, daffodils, crocuses, snowdrops)
  * Snow patches may still be visible on ground (early spring)
  * Soil looks moist, dark, freshly thawed
  * Overall color palette: bright, cool greens, pink/white blossoms
  * Trees have sparse, newly forming canopy

- AUTUMN indicators:
  * Mature leaves turning yellow, orange, red, brown
  * Fallen leaves covering the ground
  * Dried, brown or yellowish grass
  * Bare branches starting to appear (late autumn)
  * Fruits, berries, mushrooms visible
  * Overall color palette: warm golden, amber, rusty tones
  * Trees still have full canopy but changing color

USE TEMPERATURE DATA:
- If temperature is provided, COMPARE it with climate normals
- Temperatures TRENDING UP from winter norms → SPRING
- Temperatures TRENDING DOWN from summer norms → AUTUMN

USE CLIMATE NORMALS:
- Check which month's average temperature best matches the current temperature
- If current temp matches March/April/May averages → likely SPRING
- If current temp matches September/October/November averages → likely AUTUMN

MONTH NARROWING:
- Early spring (March): cold, possible snow, very few buds
- Mid spring (April): buds opening, first green appearing
- Late spring (May): full green, warm, flowers blooming
- Early autumn (September): still mostly green, slight yellowing
- Mid autumn (October): strong colors, leaves falling
- Late autumn (November): mostly bare trees, cold, brown
"""


async def analyze_photo(
    cfg: DictConfig,
    path: str,
    lat: float = None,
    lon: float = None,
    city: str = None,
    temperature: str = None,
    climate_context: str = "",
    siglip_prediction: dict = None,
) -> str:
    has_coordinates = lat is not None and lon is not None

    print("  Computing image hash...", flush=True)
    hash_val = image_hash(path)
    cache_key = (
        f"vision:{hash_val}:{lat}:{lon}:{city}:{temperature}:{hash(climate_context)}"
    )
    print("  Cache key created", flush=True)

    result = ollama_cache.get(cache_key)
    if result:
        logger.info("Vision response from cache")
        metrics.track_cache_hit()
        return result

    metrics.track_cache_miss()
    logger.info("Calling Vision model (GLM-4V-plus)")
    metrics.track_api_call("vision_model")

    # Build SigLIP hint for the prompt
    siglip_hint = ""
    if siglip_prediction:
        siglip_season = siglip_prediction.get("season", "unknown")
        siglip_conf = siglip_prediction.get("confidence", 0)
        siglip_hint = f"""
SIGLIP MODEL PREDICTION (trained on climate data, high priority):
- Predicted season: {siglip_season}
- Confidence: {siglip_conf:.2%}
- Probabilities: {json.dumps(siglip_prediction.get("probabilities", {}), ensure_ascii=False)}
Use this as a STRONG signal, especially for spring vs autumn disambiguation.
"""

    location_txt = location(lat, lon, city, temperature) if has_coordinates else ""
    if (
        climate_context
        and climate_context != "No climate data available for this location."
    ):
        climate_section = f"""
CLIMATE REFERENCE (use as additional context):
{climate_context}

Use this climate data especially to disambiguate SPRING vs AUTUMN.
Compare current temperature with monthly normals to determine which transitional season it is.
If visual evidence clearly contradicts climate data, trust VISUAL evidence.
Otherwise, use climate normals as a strong signal for transitional seasons.
"""
    else:
        climate_section = ""

    if has_coordinates:
        prompt = f"""
{location_txt}
{climate_section}

{siglip_hint}

{SEASON_SPRING_AUTUMN_GUIDE}

Analyze this image and determine the season and month.

Coordinates are available: lat={lat}, lon={lon}

STEP 1: Look at the image carefully. Note ALL visual indicators.
STEP 2: If temperature is provided ({temperature}), compare it with climate data above.
STEP 3: Combine visual evidence + temperature + climate data + SigLIP prediction to determine season.
STEP 4: Narrow down to specific month using all available data.

IMPORTANT: If unsure between spring and autumn, explicitly compare visual clues
from the guide above AND check which months' temperature norms match the current temperature.

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

Your response:"""
    else:
        prompt = f"""
{climate_section}

{siglip_hint}

{SEASON_SPRING_AUTUMN_GUIDE}

Analyze this image and determine the season and month.

NO COORDINATES AVAILABLE.
Rely ONLY on visual analysis. Pay EXTRA attention to spring vs autumn visual clues above.

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

Your response:"""

    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    try:
        response = await client.chat.completions.create(
            model=GEMINI_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        result = response.choices[0].message.content

        try:
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            parsed = json.loads(result)
            if "season" in parsed and "month" in parsed:
                if "confidence" not in parsed:
                    parsed["confidence"] = "medium"
                result = json.dumps(parsed)
                ollama_cache.set(
                    cache_key, result, ttl=cfg.model.get("cache_ttl", 3600)
                )
                return result
            else:
                logger.warning(f"Invalid response format: {result}")
                return json.dumps(
                    {"season": "unknown", "month": "unknown", "confidence": "low"}
                )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response: {result}, error: {e}")
            return json.dumps(
                {"season": "unknown", "month": "unknown", "confidence": "low"}
            )

    except Exception as e:
        logger.error(f"Vision model error: {e}", exc_info=True)
        metrics.track_error("vision_model_error")
        return json.dumps(
            {"season": "unknown", "month": "unknown", "confidence": "low"}
        )
