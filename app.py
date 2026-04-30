print("=== 1. STARTING BOT ===", flush=True)

try:
    import asyncio
    print("=== 2.1 asyncio OK ===", flush=True)
    import os
    print("=== 2.2 os OK ===", flush=True)
    import tempfile
    print("=== 2.3 tempfile OK ===", flush=True)
    import threading
    print("=== 2.4 threading OK ===", flush=True)
    from pathlib import Path
    print("=== 2.5 Path OK ===", flush=True)
    from dotenv import load_dotenv
    print("=== 2.6 dotenv OK ===", flush=True)
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    print("=== 2.7 telegram OK ===", flush=True)
    import ollama
    print("=== 2.8 ollama OK ===", flush=True)
    import hydra
    from omegaconf import DictConfig
    print("=== 2.9 hydra OK ===", flush=True)
    from flask import Flask
    print("=== 2.10 flask OK ===", flush=True)
    
    from utils import logger
    print("=== 2.11 utils.logger OK ===", flush=True)
    from utils.helpers import extract_city, parse_coordinates
    print("=== 2.12 utils.helpers OK ===", flush=True)
    from utils.geocoding import get_coordinates_by_city
    print("=== 2.13 utils.geocoding OK ===", flush=True)
    from utils.validators import validate_size
    print("=== 2.14 utils.validators OK ===", flush=True)
    from core.analyzer import analyze_photo
    print("=== 2.15 core.analyzer OK ===", flush=True)
    from graph.builder import build_agent_graph
    print("=== 2.16 graph.builder OK ===", flush=True)
    from graph.state import AgentState
    print("=== 2.17 graph.state OK ===", flush=True)
    from middleware.rate_limiter import RateLimiter
    print("=== 2.18 middleware.rate_limiter OK ===", flush=True)
    
except Exception as e:
    print(f"!!! IMPORT ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise

print("=== 2. IMPORTS DONE ===", flush=True)
load_dotenv()
print("=== 3. ENV LOADED ===", flush=True)
http_app = Flask(__name__)
print("=== 4. FLASK APP CREATED ===", flush=True)
@http_app.route('/')
def health():
    return "Season bot is running", 200

@http_app.route('/health')
def health_check():
    return {"status": "ok"}, 200

def run_http():
    port = int(os.environ.get("PORT", 10000))
    http_app.run(host="0.0.0.0", port=port, debug=False)

http_thread = threading.Thread(target=run_http, daemon=True)
http_thread.start()
print(f" HTTP server started on port {os.environ.get('PORT', 10000)}")

class SeasonBot:
    def __init__(self, cfg: DictConfig):
        print("=== 5. SEASONBOT INIT START ===", flush=True)
        self.cfg = cfg
        self.token = cfg.telegram.token
        self.ollama_host = cfg.ollama.host
        self.ollama_model = cfg.model.name

        self.ollama_client = ollama.Client(host=self.ollama_host)
        self.rate_limiter = RateLimiter(cfg)
        
        self.agent = build_agent_graph(
            cfg,
            self.ollama_client, 
            analyze_photo
        )
        
        self.application = Application.builder().token(self.token).build()
        self._register_handlers()
        
        logger.info(" SeasonBot initialized")
        logger.info(f"   Ollama: {self.ollama_host}")
        logger.info(f"   Model: {self.ollama_model}")
        print("=== 6. SEASONBOT INIT DONE ===", flush=True)
    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            " Привет! Я определяю сезон и месяц по фотографии!\n\n"
            " Отправьте фото с геолокацией или укажите город в подписи.\n\n"
            " Команды:\n"
            "/help - помощь\n"
            "/stats - статистика использования"
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            " **Справка**\n\n"
            "**Как пользоваться:**\n"
            "1. Отправьте фотографию\n"
            "2. Опционально: добавьте геолокацию или напишите город\n"
            "3. Я определю сезон и месяц\n\n"
            "**Примеры подписей:**\n"
            "• 'город Москва'\n"
            "• 'Сочи, март'\n"
            "• '55.75, 37.62'\n"
            "• '#санктпетербург'"
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from utils import metrics
        stats = metrics.get_stats()
        rate_stats = self.rate_limiter.get_stats(update.effective_user.id)
        
        reply = (
            f" **Статистика бота**\n\n"
            f" Всего запросов: {stats.get('total_requests', 0)}\n"
            f" Кэш: хиты={stats.get('cache_hits', 0)}, промахи={stats.get('cache_misses', 0)}\n"
            f" Hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%\n"
            f" Среднее время ответа: {stats.get('avg_response_time_ms', 0):.0f} мс\n"
            f" Ваши запросов: {rate_stats.get('requests_in_window', 0)}/{rate_stats.get('limit', 10)}"
        )
        await update.message.reply_text(reply)
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        allowed, wait_time = self.rate_limiter.is_allowed(user_id)
        if not allowed:
            await update.message.reply_text(f" Слишком много запросов. Подождите {wait_time} секунд.")
            return
        
        await update.message.reply_text(" Анализирую фотографию...")

        lat, lon, city = await self._extract_location(update)
        
        photo_file = await update.message.photo[-1].get_file()
        
        if photo_file.file_size > 10 * 1024 * 1024:
            await update.message.reply_text(" Фото слишком большое (максимум 10 МБ)")
            return
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
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
                has_photo=True,
                has_location=bool(lat or city),
                route=None,
                photo_analysis=None,
                photo_raw_response=None,
                rag_context=None,
                calendar_context=None,
                synthesized=None,
                answer=None,
                errors=[],
                messages=[]
            )
            
            final_state = await self.agent.ainvoke(initial_state)
            
            if final_state.get("errors"):
                await update.message.reply_text(f" {final_state['errors'][0][:100]}")
            
            if final_state.get("answer"):
                await update.message.reply_text(final_state["answer"])
            else:
                await update.message.reply_text(" Не удалось определить сезон")
                
        except Exception as e:
            logger.error(f"Error in handle_photo: {e}")
            await update.message.reply_text(" Произошла ошибка при анализе фото")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def _extract_location(self, update: Update):
        lat = None
        lon = None
        city = None
        if update.message.location:
            lat = update.message.location.latitude
            lon = update.message.location.longitude
            logger.info(f" Location from Telegram: {lat}, {lon}")
            return lat, lon, city
        caption = update.message.caption or ""
        if caption:
            coords = parse_coordinates(caption)
            if coords:
                lat, lon = coords
                logger.info(f" Coordinates from caption: {lat}, {lon}")
                return lat, lon, city
            
            city = extract_city(caption)
            if city:
                logger.info(f" City from caption: {city}")
                lat, lon = await get_coordinates_by_city(city)
                if lat and lon:
                    logger.info(f" Geocoded: {city} -> {lat}, {lon}")
        
        return lat, lon, city
    
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