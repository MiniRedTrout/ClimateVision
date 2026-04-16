import os
import json
import tempfile
import logging
from dotenv import load_dotenv
import ollama
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, filters
from flask import Flask, request
import aiohttp
from datetime import datetime
import asyncio

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден!")

# Инициализация
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Flask приложение
app = Flask(__name__)


async def get_climate_context(lat: float, lon: float) -> str:
    """Получает климатические данные через Open-Meteo API"""
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


async def analyze_photo(image_path: str, lat: float = None, lon: float = None, city: str = None) -> str:
    """Анализирует фото с учётом климатических данных"""
    climate_context = ""
    if lat and lon:
        climate_context = await get_climate_context(lat, lon)
    
    location_text = ""
    if city:
        location_text = f"Location: {city}"
    elif lat and lon:
        location_text = f"Location: {lat:.4f}, {lon:.4f}"
    
    prompt = f"""
{location_text}
{climate_context}

Analyze this image and determine the season and month.

IMPORTANT: Use climate data as PRIMARY reference.
If climate data shows March has temp above 0°C → it's SPRING, not winter.

Respond in JSON ONLY: {{"season": "...", "month": "...", "confidence": "..."}}
"""
    
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
    """Обработчик команды /start"""
    await update.message.reply_text(
        "🌍 Привет! Я определяю сезон и месяц по фотографии!\n\n"
        "📸 Отправьте фото с геолокацией или укажите город в подписи."
    )


async def handle_photo(update: Update, context):
    """Обработчик фотографий"""
    await update.message.reply_text("🔍 Анализирую фотографию...")
    
    # Получаем геолокацию
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
    
    # Скачиваем фото
    photo_file = await update.message.photo[-1].get_file()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await photo_file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    try:
        result_text = await analyze_photo(tmp_path, lat, lon, city)
        
        # Парсим JSON
        clean = result_text.strip()
        if clean.startswith('```json'):
            clean = clean[7:]
        if clean.startswith('```'):
            clean = clean[3:]
        if clean.endswith('```'):
            clean = clean[:-3]
        clean = clean.strip()
        
        result = json.loads(clean)
        
        season_ru = {"winter": "❄️ Зима", "spring": "🌸 Весна", "summer": "☀️ Лето", "autumn": "🍂 Осень"}.get(result.get("season", ""), "❓ Неизвестно")
        month_ru = {"January": "Январь", "February": "Февраль", "March": "Март", "April": "Апрель", "May": "Май", "June": "Июнь", "July": "Июль", "August": "Август", "September": "Сентябрь", "October": "Октябрь", "November": "Ноябрь", "December": "Декабрь"}.get(result.get("month", ""), "Неизвестно")
        
        answer = f"📸 Результат:\n\n🌿 Сезон: {season_ru}\n📅 Месяц: {month_ru}"
        await update.message.reply_text(answer)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:100]}")
    finally:
        os.unlink(tmp_path)


# Регистрация обработчиков
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(filters.PHOTO, handle_photo))


# Flask маршруты
@app.route('/')
def index():
    return "Season bot is running!"


@app.route(f'/webhook/{TOKEN}', methods=['POST'])
def webhook():
    """Принимает обновления от Telegram"""
    try:
        update = Update.de_json(request.get_json(), bot)
        dispatcher.process_update(update)
        return 'ok', 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return 'error', 500


# Установка вебхука при запуске
def set_webhook():
    """Устанавливает вебхук при старте приложения"""
    webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/webhook/{TOKEN}"
    
    import requests
    url = f"https://api.telegram.org/bot{TOKEN}/setWebhook"
    response = requests.post(url, json={"url": webhook_url})
    
    if response.status_code == 200:
        logger.info(f"✅ Webhook set to: {webhook_url}")
    else:
        logger.error(f"❌ Failed to set webhook: {response.text}")


if __name__ == "__main__":
    # Устанавливаем вебхук
    set_webhook()
    
    # Запускаем Flask сервер
    port = int(os.getenv("PORT", 10000))
    logger.info(f"🚀 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)