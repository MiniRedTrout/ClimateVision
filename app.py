import os
import asyncio
import json
import tempfile
import logging
from dotenv import load_dotenv
import ollama
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import aiohttp
from datetime import datetime

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден!")

ollama_client = ollama.Client(host=OLLAMA_HOST)


async def get_climate_context(lat: float, lon: float) -> str:
    """
    Получает климатические данные для координат через Open-Meteo API
    Возвращает строку с информацией для промпта
    """
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
        
        # Собираем данные по месяцам
        months_data = {}
        for i, date_str in enumerate(data["daily"]["time"]):
            month = datetime.strptime(date_str, "%Y-%m-%d").month
            temp = data["daily"]["temperature_2m_mean"][i]
            snow = data["daily"]["snowfall_sum"][i]
            
            if month not in months_data:
                months_data[month] = {"temps": [], "snow": 0}
            months_data[month]["temps"].append(temp)
            months_data[month]["snow"] += snow or 0
        
        # Формируем читаемую строку
        month_names = {
            12: "December", 1: "January", 2: "February",
            3: "March", 4: "April", 5: "May",
            6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November"
        }
        
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


async def analyze_photo_with_climate(image_path: str, lat: float = None, lon: float = None, city: str = None) -> str:
    """
    Анализирует фото с учётом климатических данных
    """
    # Получаем климатический контекст, если есть координаты
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

IMPORTANT RULES:
1. Use the climate data as the PRIMARY reference
2. If climate data shows March has avg temp above 0°C and little snow → it's SPRING, not winter
3. Only use visual clues from the image (snow, trees, clothing) as secondary evidence
4. If image shows snow but climate data says March is warming → answer "spring" and month "March"
5. If image shows snow AND climate data says December/January are cold → answer "winter"

Respond in JSON format ONLY:
{{"season": "winter/spring/summer/autumn", "month": "month_name", "confidence": "high/medium/low"}}
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


async def analyze_photo_without_climate(image_path: str) -> str:
    """
    Анализирует фото без климатических данных (если нет геолокации)
    """
    prompt = """
Analyze this image and determine the season and month.
Look for visual clues: snow, leaves, flowers, clothing, sunlight.

Respond in JSON format ONLY:
{"season": "winter/spring/summer/autumn", "month": "month_name", "confidence": "high/medium/low"}
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "🌍 Привет! Я определяю сезон и месяц по фотографии!\n\n"
        "📸 Как пользоваться:\n"
        "1. Отправьте фото\n"
        "2. Опционально: добавьте геолокацию или напишите город\n"
        "3. Я проанализирую с учётом климатических данных\n\n"
        "📍 Для точного определения месяца обязательно добавляйте геолокацию!"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик фотографий с RAG"""
    await update.message.reply_text("🔍 Анализирую фотографию с учётом климатических данных...")
    
    # Получаем геолокацию из сообщения
    lat = None
    lon = None
    city = None
    
    if update.message.location:
        lat = update.message.location.latitude
        lon = update.message.location.longitude
        logger.info(f"Got location: {lat}, {lon}")
    
    # Проверяем подпись к фото на наличие города
    if update.message.caption:
        import re
        city_match = re.search(r'(?:город|в|из)\s+([А-Яа-яA-Za-z\-]+)', update.message.caption)
        if city_match:
            city = city_match.group(1)
            logger.info(f"Got city from caption: {city}")
    
    # Скачиваем фото
    photo_file = await update.message.photo[-1].get_file()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await photo_file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Анализируем с учётом доступных данных
        if lat and lon:
            result_text = await analyze_photo_with_climate(tmp_path, lat, lon, city)
        elif city:
            # Можно добавить геокодинг города → координаты
            result_text = await analyze_photo_with_climate(tmp_path, city=city)
        else:
            result_text = await analyze_photo_without_climate(tmp_path)
        
        # Парсим и форматируем результат
        try:
            # Очищаем ответ от markdown
            clean_text = result_text.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.startswith('```'):
                clean_text = clean_text[3:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            result = json.loads(clean_text)
            
            season_ru = {
                "winter": "❄️ Зима",
                "spring": "🌸 Весна",
                "summer": "☀️ Лето",
                "autumn": "🍂 Осень"
            }.get(result.get("season", ""), "❓ Неизвестно")
            
            month_ru = {
                "January": "Январь", "February": "Февраль", "March": "Март",
                "April": "Апрель", "May": "Май", "June": "Июнь",
                "July": "Июль", "August": "Август", "September": "Сентябрь",
                "October": "Октябрь", "November": "Ноябрь", "December": "Декабрь"
            }.get(result.get("month", ""), result.get("month", "Неизвестно"))
            
            confidence_text = {
                "high": "🎯 Высокая",
                "medium": "📊 Средняя",
                "low": "🤔 Низкая"
            }.get(result.get("confidence", ""), "❓ Неизвестно")
            
            answer = f"""
📸 **Результат анализа**

🌿 **Сезон:** {season_ru}
📅 **Месяц:** {month_ru}

📊 **Уверенность:** {confidence_text}
"""
            if lat and lon:
                answer += f"\n📍 Координаты: {lat:.4f}, {lon:.4f}"
            elif city:
                answer += f"\n🏙️ Город: {city}"
            
            await update.message.reply_text(answer)
            
        except json.JSONDecodeError:
            await update.message.reply_text(f"📸 Результат: {result_text}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"❌ Ошибка при анализе: {str(e)[:100]}")
    finally:
        os.unlink(tmp_path)


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик геолокации без фото"""
    lat = update.message.location.latitude
    lon = update.message.location.longitude
    
    await update.message.reply_text(
        f"📍 Получены координаты: {lat:.4f}, {lon:.4f}\n\n"
        "Теперь отправьте фотографию места, чтобы я мог определить сезон!"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    await update.message.reply_text(
        "📸 Пожалуйста, отправьте фотографию!\n\n"
        "Я анализирую изображения и определяю сезон и месяц.\n"
        "Для помощи отправьте /start"
    )


def main():
    """Запуск бота"""
    app = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    logger.info("🚀 Бот запускается...")
    logger.info(f"📡 Подключение к Ollama: {OLLAMA_HOST}")
    
    app.run_polling()


if __name__ == "__main__":
    main()