from flask import Flask, request
from telegram import Update
import asyncio
import os
import tempfile
from pathlib import Path
from utils.logger import logger
from utils.geocoding import get_coordinates_by_city
from utils.helpers import extract_city, parse_coordinates
from graph.state import AgentState

def create_webhook_app(agent, telegram_app, main_loop, rate_limiter):
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Health check endpoint для Render"""
        return "Season bot is running!"
    
    @app.route(f'/webhook/{telegram_app.bot.token}', methods=['POST'])
    def webhook():
        try:
            json_data = request.get_json(force=True)
            update = Update.de_json(json_data, telegram_app.bot)
            if not update:
                logger.warning('Empty update')
                return 'ok', 200
            message = update.message or update.edited_message
            if not message:
                return 'ok', 200
            user_id = update.effective_user.id if update.effective_user else None
            if not user_id:
                logger.warning('No user_id')
                return 'ok', 200
            if message.text and message.text.startswith('/'):
                if message.text == '/start':
                    asyncio.run_coroutine_threadsafe(
                        send_message_async(telegram_app.bot, user_id, 
                            " Привет! Я определяю сезон и месяц по фотографии!\n\n"
                            " Отправьте фото с геолокацией или укажите город в подписи.\n\n"
                            " Команды:\n"
                            "/help - помощь\n"
                            "/stats - статистика использования"
                        ),
                        main_loop
                    )
                elif message.text == '/help':
                    asyncio.run_coroutine_threadsafe(
                        send_message_async(telegram_app.bot, user_id,
                            " **Справка**\n\n"
                            " **Как пользоваться:**\n"
                            "1. Отправьте фотографию\n"
                            "2. Опционально: добавьте геолокацию или напишите город\n"
                            "3. Я определю сезон и месяц\n\n"
                            " **Примеры подписей:**\n"
                            "• 'город Москва'\n"
                            "• 'Сочи, март'\n"
                            "• '55.75, 37.62'\n"
                            "• '#санктпетербург'"
                        ),
                        main_loop
                    )
                elif message.text == '/stats':
                    from utils import metrics
                    stats = metrics.get_stats()
                    reply = (
                        f" **Статистика бота**\n\n"
                        f" Всего запросов: {stats.get('total_requests', 0)}\n"
                        f" Кэш: хиты={stats.get('cache_hits', 0)}, промахи={stats.get('cache_misses', 0)}\n"
                        f" Hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%\n"
                        f" Среднее время ответа: {stats.get('avg_response_time_ms', 0):.0f} мс"
                    )
                    asyncio.run_coroutine_threadsafe(
                        send_message_async(telegram_app.bot, user_id, reply),
                        main_loop
                    )
                return 'ok', 200
            if rate_limiter:
                allowed, wait_time = rate_limiter.is_allowed(user_id)
                if not allowed:
                    asyncio.run_coroutine_threadsafe(
                        send_message_async(telegram_app.bot, user_id, 
                            f" Слишком много запросов. Подождите {wait_time} секунд."
                        ),
                        main_loop
                    )
                    return 'ok', 200
            if not message.photo:
                return 'ok', 200
            asyncio.run_coroutine_threadsafe(
                send_message_async(telegram_app.bot, user_id, "🔍 Анализирую фотографию..."),
                main_loop
            )
            lon = None
            city = None
            if message.location:
                lat = message.location.latitude
                lon = message.location.longitude
                logger.info(f" Location from Telegram: {lat}, {lon}")
            caption = message.caption or ""
            if caption and not (lat and lon):
                coords = parse_coordinates(caption)
                if coords:
                    lat, lon = coords
                    logger.info(f" Coordinates from caption: {lat}, {lon}")
                else:
                    city = extract_city(caption)
                    if city:
                        logger.info(f" City from caption: {city}")
            
            if city and not (lat and lon):
                lat, lon = await_async(get_coordinates_by_city(city), main_loop)
                if lat and lon:
                    logger.info(f" Geocoded: {city} -> {lat}, {lon}")
            photo_file = await_async(message.photo[-1].get_file(), main_loop)
            if photo_file.file_size > 10 * 1024 * 1024:
                asyncio.run_coroutine_threadsafe(
                    send_message_async(telegram_app.bot, user_id, "❌ Фото слишком большое (максимум 10 МБ)"),
                    main_loop
                )
                return 'ok', 200
            photos_dir = Path("temp_photos")
            photos_dir.mkdir(exist_ok=True)
            photo_path = photos_dir / f"user_{user_id}_{int(asyncio.get_event_loop().time())}.jpg"
            
            await_async(photo_file.download_to_drive(photo_path), main_loop)
            state = AgentState(
              user_id=user_id,
              user_message=caption,
              photo_path=str(photo_path),
              lat=lat,
              lon=lon,
              city=city,
              has_photo=True,
              has_location=bool(lat or city),
              photo_analysis=None,
              photo_raw_response=None,
              rag_context=None,
              synthesized=None,
              tool_result=[],
              answer=None,
              errors=[],
              messages=[]
            )
            result = await_async(agent.ainvoke(state), main_loop, timeout=90)
            if result.get("errors"):
                error_msg = f"⚠️ Частичная ошибка: {result['errors'][0][:100]}"
                asyncio.run_coroutine_threadsafe(
                    send_message_async(telegram_app.bot, user_id, error_msg),
                    main_loop
                )
            if result.get("answer"):
                asyncio.run_coroutine_threadsafe(
                    send_message_async(telegram_app.bot, user_id, result["answer"]),
                    main_loop
                )
            if photo_path.exists():
                photo_path.unlink()
            return 'ok', 200
        except asyncio.TimeoutError:
            logger.error("Handler execution timeout")
            return 'error', 500
        except Exception as e:
            logger.error(f"Webhook error: {e}", exc_info=True)
            return 'error', 500
    return app

def await_async(coro, loop, timeout=None):
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)

async def send_message_async(bot, user_id, text):
    try:
        await bot.send_message(chat_id=user_id, text=text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Failed to send message: {e}")

async def download_photo_async(photo_file, user_id):
    photos_dir = Path("temp_photos")
    photos_dir.mkdir(exist_ok=True)
    photo_path = photos_dir / f"user_{user_id}_latest.jpg"
    await photo_file.download_to_drive(photo_path)
    return str(photo_path)