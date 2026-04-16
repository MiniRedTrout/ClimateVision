import os
import asyncio
import json
import tempfile
import logging
from dotenv import load_dotenv
import ollama
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь фото, я определю сезон!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Анализирую фотографию...")
    
    photo_file = await update.message.photo[-1].get_file()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await photo_file.download_to_drive(tmp.name)
        tmp_path = tmp.name
    
    try:
        response = ollama_client.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Analyze this image and determine the season and month. Respond in JSON: {"season": "...", "month": "..."}',
                'images': [tmp_path]
            }]
        )
        result = response['message']['content']
        await update.message.reply_text(f"Результат: {result}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)[:100]}")
    finally:
        os.unlink(tmp_path)

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    logger.info("Бот запускается...")
    app.run_polling()

if __name__ == "__main__":
    main()