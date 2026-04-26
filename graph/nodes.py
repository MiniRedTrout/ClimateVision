import json 
import logging 
import re 
from datetime import datetime
from typing import Dict, Any
from .state import AgentState 
from utils.helpers import parse
from utils.logger import logger 
from utils.validators import validate_size, validate_type, validate_coords
from .tools import ALL_TOOLS
from langchain_core.messages import HumanMessage, AIMessage
from core.mcp_client import OpenMeteoMCPClient

class AgentNodes:
    def __init__(self,cfg, ollama_client,analyze_photo):
        self.ollama_client = ollama_client
        self.analyze_photo = analyze_photo
        self.cfg = cfg
        self.openmeteo = OpenMeteoMCPClient()
        self.llm_with_tools = ollama_client.bind_tools(ALL_TOOLS)
    async def router_node(self, state:AgentState)->AgentState:
        logger.info("Router:анализирует")
        if not state.get('messages'):
            state['messages'] = []
        state['has_photo'] = bool(state.get('photo_path'))
        state['has_location'] = bool(
            (state.get('lat') and state.get('lon')) or state.get('city')
        )
        if state.get('lat') and state.get('lon'):
            is_valid, error = validate_coords(state['lat'], state['lon'])
            if not is_valid:
                logger.warning(f"Invalid coordinates: {error}")
                state['errors'].append(error)
                state['has_location'] = False
        if state.get('user_message'):
            state['messages'].append(HumanMessage(content=state['user_message']))
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
            climate_context = state.get('rag_context', '')
            result = await self.analyze_photo(
                self.cfg,
                state['photo_path'],
                state.get('lat'),
                state.get('lon'),
                state.get('city'),
                self.ollama_client,
                climate_context
            )
            state["photo_raw_response"] = result
            state["photo_analysis"] = parse(result)
            state['messages'].append({
               "role": "assistant",
               "content": f"Photo analysis complete: season={state['photo_analysis'].get('season')}, month={state['photo_analysis'].get('month')}",
               "timestamp": datetime.now().isoformat(),
               "type": "photo_analysis"
            })

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
    async def climate_node(self,state:AgentState)->AgentState:
        logger.info('Climate node')
        if state.get('rag_context'):
            return state 
        if state.get('lat') and state.get('lon'):
            try:
                climate_mcp = await self.openmeteo.get_climate_history(
                    state['lat'], 
                    state['lon']
                )
                state['rag_context'] = climate_mcp
                logger.info(f"Climate context retrieved")
            except Exception as e:
                logger.error(f"Climate MCP error: {e}")
    
        elif state.get('city'):
            try:
                climate_mcp = await self.openmeteo.get_city_climate(state['city'])
                state['rag_context'] = climate_mcp
            
            except Exception as e:
                logger.error(f"City climate error: {e}")
    
        return state
    async def synthesis_node(self,state:AgentState)->AgentState:
        logger.info("Synthesis node")
        photo = state.get('photo_analysis',{})
        climate = state.get('rag_context', '')
        user_message = state.get('user_message', '')
        state['synthesized'] = {
            'season': photo.get('season', 'unknown'),
            'month': photo.get('month', 'unknown'),
            'confidence': photo.get('confidence', 'medium')
        }
        prompt = f"""
Ты помощник, который определяет сезон по фотографии.

Данные с фото:
- Сезон: {photo.get('season', 'unknown')}
- Месяц: {photo.get('month', 'unknown')}
- Уверенность: {photo.get('confidence', 'medium')}

Климатический контекст:
{climate}

Вопрос пользователя: {user_message}

Если пользователь спрашивает о погоде, климате или сравнении городов - используй доступные инструменты.
Ответь полезно и дружелюбно.
"""
        messages = state.get('messages', [])
        messages.append(HumanMessage(content=prompt))
        response = await self.llm_with_tools.ainvoke(messages)
        messages.append(response)
        state['messages'] = messages
        state['last_llm_response'] = response
        logger.info(f"LLM response has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
        
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
    