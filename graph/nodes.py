import json 
import logging 
import re 
from datetime import datetime
from typing import Dict, Any
from .state import AgentState 
from utils.helpers import parse
from utils.logger import logger 
from utils.validators import validate_size, validate_type, validate_coords

class AgentNodes:
    def __init__(self,cfg, ollama_client, climate_retriever,analyze_photo):
        self.ollama_client = ollama_client
        self.climate_retriever = climate_retriever
        self.analyze_photo = analyze_photo
        self.cfg = cfg
        self._siglip = None
    def router_node(self, state:AgentState)->AgentState:
        logger.info("Router:анализирует")
        state['has_photo'] = bool(state.get('photo_path'))
        state['has_location'] = bool(
            (state.get('lat') and state.get('lon')) or state.get('city')
        )
        if state.get('lat') and state.get('lon'):
            is_valid, error = validate_coords(state['lat'], state['lon'])
            if not is_valid:
                logger.warning(f"Invalid coordinates: {error}")
                state['errors'] = state.get('errors', []) + [error]
                state['has_location'] = False
        logger.info(f'Photo: {state['has_photo']}')
        logger.info(f'Location: {state['has_location']}')
        return state 
    async def analysis_node(self,state:AgentState)->AgentState:
        logger.info('Photo Analysis')
        if not state.get('photo_path'):
            logger.warning('Нет фото')
            return state 
        valid_size,size_error = validate_size(state['photo_path'])
        if not valid_size:
            logger.error(size_error)
            state['errors'].append(size_error)
            return state
        valid_type,type_error = validate_type(state['photo_path'])
        if not valid_type:
            logger.error(type_error)
            state['errors'].append(type_error)
            return state
        try:
            result = await self.analyze_photo(
                self.cfg,
                state['photo_path'],
                state.get('lat'),
                state.get('lon'),
                state.get('city'),
                self.ollama_client
            )
            state["photo_raw_response"] = result
            state["photo_analysis"] = parse(result)
            logger.info(f"Результат: {state['photo_analysis']}")
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            state["errors"].append(str(e))
            state["photo_analysis"] = {
                "season": "unknown",
                "month": "unknown",
                "confidence": "low"
            }
        
        return state
    async def siglip_node(self,state:AgentState)->AgentState:
        logger.info('SigLIP node')
        if not hasattr(self,'siglip'):
            from core.siglip import siglip
            self.siglip = siglip 
        match = self.siglip.find_similar(state['photo_path'])
        if match and match.get('similarity', 0) > 0.85:
            state['siglip_match'] = match
            state['photo_analysis'] = {
            "season": match['season'],
            "month": match['month'],
            "confidence": "high",
            "source": "siglip"
            }
            logger.info(f"SigLIP match: {match['season']}/{match['month']} (sim: {match['similarity']:.2f})")
        else:
            logger.info("SigLIP: похожих эталонов не найдено")
        return state
    def synthesis_node(self,state:AgentState)->AgentState:
        logger.info("Synthesis node")
        photo = state.get('photo_analysis',{})
        season = photo.get('season','unknown')
        month = photo.get('month', 'unknown')
        confidence = photo.get('confidence', 'medium')
        state['synthesized'] = {
            'season': season,
            'month': month,
            'confidence': confidence
        }
        logger.info(f"Итог: сезон={season}, месяц={month}, уверенность={confidence}")
        return state
    def formatter_node(self, state: AgentState) -> AgentState:
        logger.info("Formatter Node")
        synthesized = state.get('synthesized', {})
        season_ru = self.cfg.graph.SEASON_NAMES_RU.get(synthesized.get('season', 'unknown'), '❓ Неизвестно')
        month_ru = self.cfg.graph.MONTH_NAMES_RU.get(synthesized.get('month', ''), 'Неизвестно')

        confidence_icon = {
            'high': '',
            'medium': '',
            'low': ''
        }.get(synthesized.get('confidence', 'medium'), '')
        
        state['answer'] = f"""
**Результат анализа**

Сезон: {season_ru}
Месяц: {month_ru}

{confidence_icon} Уверенность: {synthesized.get('confidence', 'medium')}
"""
        sources = []
        if state.get('photo_analysis') and state['photo_analysis'].get('season') != 'unknown':
            sources.append('анализ фото')
        if sources:
            state['answer'] += f"\nИсточники: {', '.join(sources)}"
        
        return state
    