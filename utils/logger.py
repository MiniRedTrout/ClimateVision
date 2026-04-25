import logging 
import sys 
from typing import Optional 
def setup_logger(
        name: str= __name__,
        level: int= logging.INFO,
        log_file: Optional[str] = None
)->logging.Logger:
    """Настраиваем логгер"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
logger = setup_logger('season_bot')
