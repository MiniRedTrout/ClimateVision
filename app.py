import threading
import time
import logging
from dotenv import load_dotenv
from flask import Flask
from telegram import Bot
from telegram.ext import Application
import ollama
import hydra
import asyncio 
from omegaconf import DictConfig, OmegaConf
from utils import logger
from core.analyzer import analyze_photo
from core.climate import climate_retriever
from graph.builder import build_agent_graph
from bot import BotHandlers, create_webhook_app
from middleware.rate_limiter import RateLimiter
from middleware.error_handler import handle_errors
load_dotenv()

main_loop = None
telegram_app = None
agent = None

def init_global_loop():
    global main_loop, telegram_app, agent
    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)
    telegram_app = Application.builder().token(cfg_global.telegram.token).build()
    main_loop.run_until_complete(telegram_app.initialize())
    main_loop.run_until_complete(telegram_app.bot.initialize())
    logger.info(" Telegram application initialized")
    
    ollama_client = ollama.Client(host=cfg_global.ollama.host)
    agent = build_agent_graph(ollama_client, climate_retriever, analyze_photo)
    logger.info("Agent graph built")

    try:
        main_loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        main_loop.close()

def start_loop_in_thread():
    thread = threading.Thread(target=init_global_loop, daemon=True)
    thread.start()
    time.sleep(3) 
    logger.info("Event loop running in background thread")


def set_webhook():
    if not cfg_global.telegram.webhook_host or cfg_global.telegram.webhook_host == "localhost":
        logger.warning("Webhook host not set, skipping webhook setup")
        return
    webhook_url = f"https://{cfg_global.telegram.webhook_host}/webhook/{cfg_global.telegram.token}"
    import requests
    url = f"https://api.telegram.org/bot{cfg_global.telegram.token}/setWebhook"
    response = requests.post(url, json={"url": webhook_url})
    if response.status_code == 200 and response.json().get('ok'):
        logger.info(f"Webhook set to: {webhook_url}")
    else:
        logger.error(f"Failed to set webhook: {response.text}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    global cfg_global
    cfg_global = cfg
    logger.info("Starting Season Bot...")
    logger.info(f"   Ollama host: {cfg.ollama.host}")
    logger.info(f"   Model: {cfg.model.name}")
    logger.info(f"   Port: {cfg.telegram.port}")
    rate_limiter = RateLimiter(cfg)
    start_loop_in_thread()
    set_webhook()
    flask_app = create_webhook_app(agent, telegram_app, main_loop,rate_limiter)
    logger.info(f" Starting Flask server on port {cfg.telegram.port}")
    flask_app.run(host="0.0.0.0", port=cfg.telegram.port)

if __name__ == "__main__":
    main()

