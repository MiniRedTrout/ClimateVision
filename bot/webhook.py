
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
            if not update:
                logger.warning('Empty update')
                return 'ok',200
            message = update.message or update.edited_message
            if not message:
                return 'ok',200
            user_id = update.effective_user.id if update.effective_user else None 
            if not user_id:
                logger.warning('No user_id')
                return 'ok',200 
            user_message = message.text if message.text else None 
            has_photo = bool(message.photo)
            has_location = bool(message.location)
            lat = None 
            lon = None 
            city = None 
            if has_location:
                lat = message.location.latitude
                lon = message.location.longitude
            elif message.text and ',' in message.text:
                parts = message.text.split(',')
                if len(parts) == 2:
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                    except ValueError:
                        pass 
            state = {
                "user_id": user_id,
                "user_message": user_message,
                "photo_path": None,  
                "lat": lat,
                "lon": lon,
                "city": city,
                "has_photo": has_photo,
                "has_location": has_location,
                "photo_analysis": None,
                "rag_context": None,
                "calendar_context": None,
                "synthesized": None,
                "answer": None,
                "errors": [],
                "messages": []
            }
            if has_photo:
                try:
                    photo_file = message.photo[-1].get_file()
                    future = asyncio.run_coroutine_threadsafe(
                        download_photo_async(photo_file, user_id),
                        main_loop
                    )
                    photo_path = future.result(timeout=30)
                    state["photo_path"] = photo_path
                except Exception as e:
                    logger.error(f"Failed to download photo: {e}")
                    state["errors"].append(f"Photo download failed: {str(e)}")
            
            future = asyncio.run_coroutine_threadsafe(
                agent.ainvoke(state),
                main_loop
            )
            result = future.result(timeout=60)
            if result.get("answer"):
                asyncio.run_coroutine_threadsafe(
                    send_message_async(telegram_app.bot, user_id, result["answer"]),
                    main_loop
                )
            
            return 'ok', 200
            
        except asyncio.TimeoutError:
            logger.error("Agent execution timeout")
            return 'error', 500
        except Exception as e:
            logger.error(f"Webhook error: {e}", exc_info=True)
            return 'error', 500
    
    return app

async def download_photo_async(photo_file, user_id):
    import os
    from pathlib import Path
    photos_dir = Path("temp_photos")
    photos_dir.mkdir(exist_ok=True)
    photo_path = photos_dir / f"user_{user_id}_latest.jpg"
    await photo_file.download_to_drive(photo_path)
    
    return str(photo_path)

async def send_message_async(bot, user_id, text):
    try:
        await bot.send_message(chat_id=user_id, text=text)
    except Exception as e:
        logger.error(f"Failed to send message: {e}")