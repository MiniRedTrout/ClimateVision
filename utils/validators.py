from typing import Tuple 
from pathlib import Path
from omegaconf import OmegaConf
def validate_size(path:str,cfg:OmegaConf)->Tuple[bool,str]:
    """Проверит фото размер"""
    size = Path(path).stat().st_size / (1024 * 1024)
    if size > cfg.photo.max_size:
        return False, f'Фото большое - {size:.1f} МБ. Максимум - {cfg.photo.max_size} МБ.'
    return True, ''

def validate_type(path:str,cfg:OmegaConf)->Tuple[bool,str]:
    """Проверит тип фото"""
    extention = path.split('.')[-1].lower()
    if extention not in cfg.photo.types:
        return False,f'Неподдерживается этот формат фото, используйте - {cfg.photo.types}.'

def validate_coords(lat: float, lon: float)->Tuple[bool,str]:
    """Проверит координаты"""
    if not (-90 <= lat <= 90):
        return False, f'Невозможная широта {lat}.'
    if not (-180 <= lon <= 180):
        return False, f"Невозможная долгота: {lon}."
    return True, ""
