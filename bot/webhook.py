
from flask import Flask, request
from telegram import Update
import asyncio
from middleware import error_handler
from utils.logger import logger

def create_webhook_app(agent, telegram_app, main_loop):
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return "Season bot is running!"
    @app.route(f'/webhook/{telegram_app.bot.token}', methods=['POST'])
    def webhook():
        try:
            json_data = request.get_json(force=True)
            update = Update.de_json(json_data, telegram_app.bot)
            future = asyncio.run_coroutine_threadsafe(
                agent.ainvoke({
                    "user_id": update.effective_user.id if update.effective_user else None,
                    "user_message": update.message.text if update.message else None,
                    "photo_path": None,
                    "lat": None,
                    "lon": None,
                    "city": None,
                    "has_photo": bool(update.message.photo) if update.message else False,
                    "has_location": bool(update.message.location) if update.message else False,
                    "photo_analysis": None,
                    "rag_context": None,
                    "calendar_context": None,
                    "synthesized": None,
                    "answer": None,
                    "errors": [],
                    "messages": []
                }),
                main_loop
            )
            result = future.result(timeout=60)
            return 'ok', 200
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return 'error', 500
    return app