import traceback
from functools import wraps
from typing import Callable, Any
from utils import logger 

class ErrorHandler:
    """Обрабатывает ошибки"""
    @staticmethod
    def handle_ollama_error(e: Exception)->str:
        logger.error(f"Ollama error: {e}")
        return "Ошибка подключения к модели"
    @staticmethod
    def handle_telegram_error(e: Exception)->str:
        logger.error(f"Telegram error: {e}")
        return "Ошибка при отправке сообщения."
    @staticmethod
    def handle_api_error(e: Exception) -> str:
        logger.error(f"API error: {e}")
        return "Ошибка получения климатических данных"
    @staticmethod
    def handle_general_error(e: Exception) -> str:
        logger.error(f"General error: {traceback.format_exc()}")
        return "Произошла неизвестная ошибка"

def handle_errors(func:Callable)->Callable:
    """Декоратор"""
    @wraps(func)
    async def wrapper(*args,**kwargs):
        try:
            return await func(*args,**kwargs)
        except Exception as e:
            error_handler = ErrorHandler()
            user_message = error_handler.handle_general_error(e)
            for arg in args:
                if hasattr(arg,'message') and hasattr(arg.message,'reply_text'):
                    await arg.message.reply_text(user_message)
                    break 
            return None 
    return wrapper 

error_handler = ErrorHandler()