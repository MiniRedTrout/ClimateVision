import os
import json
import tempfile
import logging
from dotenv import load_dotenv
import ollama
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from flask import Flask, request
import aiohttp
from datetime import datetime
import asyncio
from rag.retriever import ClimateRetriever
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден!")

app = Flask(__name__)

bot = Bot(token=TOKEN)
telegram_app = Application.builder().token(TOKEN).build()
ollama_client = ollama.Client(host=OLLAMA_HOST)

initialized_app = None


def init_telegram_app():
    global initialized_app
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(telegram_app.initialize())
    loop.close()
    initialized_app = telegram_app
    logger.info("✅ Telegram application initialized")

async def get_climate_context_api(lat: float, lon: float) -> str:
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "daily": ["temperature_2m_mean", "snowfall_sum"],
            "timezone": "auto"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
        
        if "daily" not in data:
            return ""
        
        months_data = {}
        for i, date_str in enumerate(data["daily"]["time"]):
            month = datetime.strptime(date_str, "%Y-%m-%d").month
            temp = data["daily"]["temperature_2m_mean"][i]
            snow = data["daily"]["snowfall_sum"][i]
            
            if month not in months_data:
                months_data[month] = {"temps": [], "snow": 0}
            months_data[month]["temps"].append(temp)
            months_data[month]["snow"] += snow or 0
        
        month_names = {12: "December", 1: "January", 2: "February", 3: "March", 4: "April"}
        context = "\n📊 Climate data for this location (based on 2023):\n"
        for month in [12, 1, 2, 3, 4]:
            if month in months_data:
                avg_temp = sum(months_data[month]["temps"]) / len(months_data[month]["temps"])
                snow = months_data[month]["snow"]
                context += f"   • {month_names[month]}: {avg_temp:.1f}°C, snow {snow:.0f}mm\n"
        
        return context
    except Exception as e:
        logger.warning(f"Climate API error: {e}")
        return ""
climate_retriever = ClimateRetriever()
async def get_climate_context_hybrid(lat:float=None,lon:float = None,city: str = None):
    context = climate_retriever.get_climate_context(lat,lon,city)
    if context:
        logger.info('Using RAG')
        return context 
    if lat and lon:
        logger.info('RAG not found, using API Open-Meteo')
        return await get_climate_context_api(lat,lon)
    return ''

async def analyze_photo(image_path: str, lat: float = None, lon: float = None, city: str = None) -> str:
    climate_context = await get_climate_context_hybrid(lat, lon,city)
    
    location_text = ""
    if city:
        location_text = f"Location: {city}"
    elif lat and lon:
        location_text = f"Location: {lat:.4f}, {lon:.4f}"
    
    prompt = f"""
{location_text}
{climate_context}

Analyze this image. You MUST determine BOTH season AND month.
If you cannot determine, use "unknown" for season and "unknown" for month.

Possible seasons: winter, spring, summer, autumn
Possible months: January, February, March, April, May, June, July, August, September, October, November, December

Respond ONLY with valid JSON. No other text.
Example: {{"season": "winter", "month": "December", "confidence": "high"}}

Your response:"""
    
    response = ollama_client.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }]
    )
    
    return response['message']['content']


async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Я определяю сезон и месяц по фотографии!\n\n"
        "Отправьте фото с геолокацией или укажите город в подписи."
    )


async def handle_photo(update: Update, context):
    await update.message.reply_text(" Анализирую фотографию...")
    
    lat = None
    lon = None
    city = None
    
    if update.message.location:
        lat = update.message.location.latitude
        lon = update.message.location.longitude
    
    if update.message.caption:
        import re
        city_match = re.search(r'(?:город|в|из)\s+([А-Яа-яA-Za-z\-]+)', update.message.caption)
        if city_match:
            city = city_match.group(1)
    
    photo_file = await update.message.photo[-1].get_file()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await photo_file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    try:
      result_text = await analyze_photo(tmp_path, lat, lon, city)
    
      logger.info(f"Raw response: {result_text}")
      clean = result_text.strip()
      if clean.startswith('```json'):
        clean = clean[7:]
      if clean.startswith('```'):
        clean = clean[3:]
      if clean.endswith('```'):
        clean = clean[:-3]
      clean = clean.strip()
    
      result = json.loads(clean)
    
      if result.get("season") == "unknown" and result.get("month") != "unknown":
        month_to_season = {
            "December": "winter", "January": "winter", "February": "winter",
            "March": "spring", "April": "spring", "May": "spring",
            "June": "summer", "July": "summer", "August": "summer",
            "September": "autumn", "October": "autumn", "November": "autumn"
        }
        month = result.get("month")
        if month in month_to_season:
            result["season"] = month_to_season[month]
            logger.info(f"Auto-corrected season to {result['season']}")
    
      if result.get("season") == "unknown" and result.get("month") == "unknown":
        await update.message.reply_text("❌ Не удалось определить сезон и месяц по этому фото. Попробуйте другое фото или добавьте геолокацию.")
        return
    
      season_ru = {"winter": "❄️ Зима", "spring": "🌸 Весна", "summer": "☀️ Лето", "autumn": "🍂 Осень"}.get(result.get("season", ""), "❓ Неизвестно")
      month_ru = {"January": "Январь", "February": "Февраль", "March": "Март", "April": "Апрель", "May": "Май", "June": "Июнь", "July": "Июль", "August": "Август", "September": "Сентябрь", "October": "Октябрь", "November": "Ноябрь", "December": "Декабрь"}.get(result.get("month", ""), "Неизвестно")
    
      answer = f"📸 Результат:\n\n🌿 Сезон: {season_ru}\n📅 Месяц: {month_ru}"
      await update.message.reply_text(answer)
    
    except json.JSONDecodeError as e:
      logger.error(f"JSON parse error: {e}, response: {result_text}")
      await update.message.reply_text(f"❌ Ошибка обработки ответа от модели. Техническая проблема.")
    except Exception as e:
      logger.error(f"Error: {e}")
      await update.message.reply_text(f"❌ Ошибка: {str(e)[:100]}")


telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))


@app.route('/')
def index():
    return "Season bot is running!"


@app.route(f'/webhook/{TOKEN}', methods=['POST'])
def webhook():
    global initialized_app
    
    try:
        json_data = request.get_json(force=True)
        update = Update.de_json(json_data, bot)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(initialized_app.process_update(update))
        loop.close()
        
        return 'ok', 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return 'error', 500


def set_webhook():
    host = os.getenv('RENDER_EXTERNAL_HOSTNAME')
    if not host:
        logger.warning("RENDER_EXTERNAL_HOSTNAME not set, using localhost")
        webhook_url = f"https://localhost/webhook/{TOKEN}"
    else:
        webhook_url = f"https://{host}/webhook/{TOKEN}"
    
    import requests
    url = f"https://api.telegram.org/bot{TOKEN}/setWebhook"
    response = requests.post(url, json={"url": webhook_url})
    
    if response.status_code == 200 and response.json().get('ok'):
        logger.info(f"✅ Webhook set to: {webhook_url}")
    else:
        logger.error(f"❌ Failed to set webhook: {response.text}")


if __name__ == "__main__":
    init_telegram_app()
    
    set_webhook()
    
    port = int(os.getenv("PORT", 10000))
    logger.info(f" Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)