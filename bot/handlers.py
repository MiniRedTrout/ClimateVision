import os 
import tempfile 
from telegram import Update
from telegram.ext import ContextTypes
from utils.helpers import extract_city, validate_size, validate_type
from middleware import rate_limiter
from utils.logger import logger
import sys
import asyncio


class BotHandlers:
    def __init__(self, agent, rate_limiter):
        self.agent = agent
        self.rate_limiter = rate_limiter
    
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
            "**Справка**\n\n"
            "**Как пользоваться:**\n"
            "1. Отправьте фотографию\n"
            "2. Опционально: добавьте геолокацию или напишите город\n"
            "3. Я определю сезон и месяц\n\n"
            "**Примеры подписей:**\n"
            "- 'город Москва'\n"
            "- 'Сочи, март'\n"
            "- '#санктпетербург'\n\n"
            "Бот использует климатическую базу данных 160+ городов мира."
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from utils import metrics
        stats = metrics.get_stats()
        rate_stats = {}
        if self.rate_limiter:
            rate_stats = self.rate_limiter.get_stats(update.effective_user.id)
        reply = (
            f" **Статистика бота**\n\n"
            f" Всего запросов: {stats.get('total_requests', 0)}\n"
            f" Кэш: хиты={stats.get('cache_hits', 0)}, промахи={stats.get('cache_misses', 0)}\n"
            f" Hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%\n"
            f" Среднее время ответа: {stats.get('avg_response_time_ms', 0):.0f} мс\n"
        )
        
        if rate_stats:
            reply += f"\n🚦 Ваши запросов: {rate_stats.get('requests_in_window', 0)}/{rate_stats.get('limit', 10)}"
        
        await update.message.reply_text(reply)
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id

        if self.rate_limiter:
            allowed, wait_time = self.rate_limiter.is_allowed(user_id)
            if not allowed:
                await update.message.reply_text(
                    f" Слишком много запросов. Подождите {wait_time} секунд."
                )
                return
        await update.message.reply_text(" Анализирую фотографию...")
        lat = None
        lon = None
        city = None
        if update.message.location:
            lat = update.message.location.latitude
            lon = update.message.location.longitude
            logger.info(f" Location: {lat}, {lon}")
        if update.message.caption:
            city = extract_city(update.message.caption)
            if city:
                logger.info(f" City: {city}")
        photo_file = await update.message.photo[-1].get_file()
        if photo_file.file_size > 10 * 1024 * 1024:
            await update.message.reply_text(" Фото слишком большое (максимум 10 МБ)")
            return
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            await photo_file.download_to_drive(tmp.name)
            tmp_path = tmp.name
        try:
            from graph import AgentState
            
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
                await update.message.reply_text(
                    f" Частичная ошибка: {final_state['errors'][0][:100]}"
                )
            
            await update.message.reply_text(final_state["answer"])
            
        except Exception as e:
            logger.error(f"Error in handle_photo: {e}")
            await update.message.reply_text(" Произошла ошибка при анализе фото")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)