"""
ClimateVision — Telegram-бот для определения сезона и месяца по фотографии.
Принимает фото + геолокацию/город/температуру.
Использует GLM-4V (vision) для анализа изображения,
климатические данные (RAG + OpenMeteo API) для валидации,
LangGraph pipeline для обработки запроса.
"""

print("START", flush=True)


from dotenv import load_dotenv

load_dotenv()

import asyncio
import os
import tempfile
import threading
import time

import hydra
from flask import Flask
from omegaconf import DictConfig
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from utils import logger

print("Дошли до logger", flush=True)
from core.analyzer import analyze_photo
from utils.geocoding import get_coordinates_by_city
from utils.helpers import extract_city, extract_temperature, parse_coordinates
from utils.validators import validate_coords

print("Перед графом", flush=True)
try:
    from graph.builder import build_agent_graph
except Exception as e:
    print(f"!!! ERROR importing graph.builder: {e}", flush=True)
    raise
from graph.state import AgentState
from middleware.rate_limiter import RateLimiter

http_app = Flask(__name__)


@http_app.route("/")
def health():
    return "Season bot is running", 200


@http_app.route("/health")
def health_check():
    return {"status": "ok"}, 200


def run_http():
    port = int(os.environ.get("PORT", 10000))
    http_app.run(host="0.0.0.0", port=port, debug=False)


http_thread = threading.Thread(target=run_http, daemon=True)
http_thread.start()

print("=== 2. IMPORTS DONE ===", flush=True)
print("=== 3. ENV LOADED ===", flush=True)
print("=== 4. FLASK APP CREATED ===", flush=True)
print(f" HTTP server started on port {os.environ.get('PORT', 10000)}")


class SeasonBot:
    def __init__(self, cfg: DictConfig):
        print("=== 5. SEASONBOT INIT START ===", flush=True)
        self.cfg = cfg
        self.token = cfg.telegram.token
        self.rate_limiter = RateLimiter(cfg)

        self.agent = build_agent_graph(cfg, analyze_photo)

        self.application = Application.builder().token(self.token).build()
        self._register_handlers()

        logger.info("SeasonBot initialized")
        print("=== 6. SEASONBOT INIT DONE ===", flush=True)

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Привет! Я определяю сезон и месяц по фотографии!\n\n"
            "Отправьте фото с геолокацией или укажите город в подписи.\n\n"
            "Команды:\n"
            "/help - помощь\n"
            "/stats - статистика использования"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Справка\n\n"
            "Как пользоваться:\n"
            "1. Отправьте фотографию\n"
            "2. Опционально: добавьте геолокацию или напишите город\n"
            "3. Я определю сезон и месяц\n\n"
            "Примеры подписей:\n"
            "• 'город Москва'\n"
            "• 'Сочи, март'\n"
            "• '55.75, 37.62'\n"
            "• '#санктпетербург'\n"
            "• 'температура +5°C'"
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from utils import metrics

        stats = metrics.get_stats()
        rate_stats = self.rate_limiter.get_stats(update.effective_user.id)

        reply = (
            f"Статистика бота\n\n"
            f"Всего запросов: {stats.get('total_requests', 0)}\n"
            f"Кэш: хиты={stats.get('cache_hits', 0)}, "
            f"промахи={stats.get('cache_misses', 0)}\n"
            f"Hit rate: {stats.get('cache_hit_rate', 0) * 100:.1f}%\n"
            f"Среднее время ответа: "
            f"{stats.get('avg_response_time_ms', 0):.0f} мс\n"
            f"Ваших запросов: "
            f"{rate_stats.get('requests_in_window', 0)}/"
            f"{rate_stats.get('limit', 10)}"
        )
        await update.message.reply_text(reply)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from utils import metrics

        metrics.track_request()
        start_time = time.time()
        user_id = update.effective_user.id

        allowed, wait_time = self.rate_limiter.is_allowed(user_id)
        if not allowed:
            await update.message.reply_text(
                f"Слишком много запросов. Подождите {wait_time} секунд."
            )
            return

        await update.message.reply_text("Анализирую фотографию...")

        lat, lon, city, temperature = await self._extract_location(update)

        feedback_parts = ["Принял данные:"]
        warnings = []

        if city:
            feedback_parts.append(f"Город: {city}")
        if lat and lon:
            valid, coord_error = validate_coords(lat, lon)
            if valid:
                feedback_parts.append(f"Координаты: {lat:.4f}, {lon:.4f}")
            else:
                warnings.append(f"{coord_error} - исправьте и отправьте заново")
        if temperature is not None:
            if -60 <= temperature <= 60:
                feedback_parts.append(f"Температура: {temperature}°C")
            else:
                warnings.append(
                    f"Странная температура: {temperature}°C. "
                    f"Проверьте формат (например: +5°C, -10°C)"
                )

        if warnings:
            feedback_parts.append("")
            feedback_parts.extend(warnings)

        if not city and not (lat and lon):
            feedback_parts.append("")
            feedback_parts.append(
                "Подсказка: укажите город или координаты для лучшего результата."
            )
            feedback_parts.append('Примеры: "город Москва", "55.75, 37.62", "+5°C"')

        await update.message.reply_text("\n".join(feedback_parts))

        photo_file = await update.message.photo[-1].get_file()

        if photo_file.file_size > 10 * 1024 * 1024:
            await update.message.reply_text("Фото слишком большое (максимум 10 МБ)")
            return

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            await photo_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        try:
            initial_state = AgentState(
                user_id=user_id,
                user_message=update.message.caption or "",
                photo_path=tmp_path,
                lat=lat,
                lon=lon,
                city=city,
                temperature=temperature,
                has_photo=True,
                has_location=bool(lat or city),
                route=None,
                photo_analysis=None,
                photo_raw_response=None,
                rag_context=None,
                last_llm_response=None,
                synthesized=None,
                tool_result=[],
                answer=None,
                errors=[],
                messages=[],
            )

            print("=== INITIAL STATE CREATED ===", flush=True)
            final_state = await self.agent.ainvoke(initial_state)
            print("=== GRAPH EXECUTED OK ===", flush=True)

            if final_state.get("errors"):
                await update.message.reply_text(f"{final_state['errors'][0][:100]}")

            if final_state.get("answer"):
                await update.message.reply_text(final_state["answer"])
            else:
                await update.message.reply_text("Не удалось определить сезон")
            duration_ms = (time.time() - start_time) * 1000
            metrics.track_response_time(duration_ms)
        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            print(f"!!! ПОЛНЫЙ ТРЕЙСБЭК:\n{error_detail}", flush=True)
            logger.error(f"Error in handle_photo: {e}", exc_info=True)
            await update.message.reply_text("Произошла ошибка при анализе фото")

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _extract_location(self, update: Update):
        lat = None
        lon = None
        city = None
        temperature = None
        if update.message.location:
            lat = update.message.location.latitude
            lon = update.message.location.longitude
            logger.info(f"Location from Telegram: {lat}, {lon}")
        caption = update.message.caption or ""
        if caption:
            print(caption, flush=True)
            coords = parse_coordinates(caption)
            if coords:
                lat, lon = coords
                logger.info(f"Coordinates from caption: {lat}, {lon}")

            city = extract_city(caption)
            if city:
                logger.info(f"City from caption: {city}")
                lat, lon = await get_coordinates_by_city(city)
                if lat and lon:
                    logger.info(f"Geocoded: {city} -> {lat}, {lon}")
            temperature = extract_temperature(caption)
            if temperature:
                logger.info(f"Temperature from caption: {temperature}")
            else:
                logger.info(f"Temperature NOT found in caption: '{caption}'")
        return lat, lon, city, temperature

    async def run(self):
        logger.info("Starting bot in polling mode...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        try:
            while True:
                await asyncio.sleep(3600)
                logger.info("Bot is alive")
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.application.stop()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("=== 7. MAIN START ===", flush=True)
    logger.info("Starting Season Bot Worker...")

    bot = SeasonBot(cfg)
    print("=== 8. BOT CREATED ===", flush=True)
    asyncio.run(bot.run())
    print("=== 9. BOT RUN DONE ===", flush=True)


print("=== 10. BEFORE MAIN CALL ===", flush=True)
if __name__ == "__main__":
    main()
    print("=== 11. AFTER MAIN ===", flush=True)
