from datetime import datetime

from .state import AgentState

print(1, flush=True)
print(2, flush=True)
from utils.helpers import parse

print(3, flush=True)
from utils.logger import logger

print(4, flush=True)
from utils.validators import validate_coords, validate_size, validate_type

print(5, flush=True)
print(6, flush=True)
print("Node", flush=True)
import json
import os

import openai

from core.siglip import SigLIPClassifier

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"

groq_client = openai.AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_BASE,
)


class AgentNodes:
    def __init__(self, cfg, analyze_photo):
        self.analyze_photo = analyze_photo
        self.cfg = cfg
        self._climate_retriever = None
        self._siglip = None

    def get_siglip(self):
        if self._siglip is None:
            try:
                self._siglip = SigLIPClassifier.get_instance()
                logger.info("SigLIP модель загружена")
            except Exception as e:
                logger.error(f"Не удалось загрузить SigLIP: {e}")
                self._siglip = None
        return self._siglip

    async def router_node(self, state: AgentState) -> AgentState:
        logger.info("Router: анализирует")
        if not state.get("messages"):
            state["messages"] = []
        state["has_photo"] = bool(state.get("photo_path"))
        state["has_location"] = bool(
            (state.get("lat") and state.get("lon")) or state.get("city")
        )
        if state.get("lat") and state.get("lon"):
            is_valid, error = validate_coords(state["lat"], state["lon"])
            if not is_valid:
                logger.warning(f"Invalid coordinates: {error}")
                state["errors"].append(error)
                state["has_location"] = False
        if state.get("user_message"):
            state["messages"].append({"role": "user", "content": state["user_message"]})
        return state

    def get_retriever(self):
        logger.info("get_retriever: START")
        if self._climate_retriever is None:
            from rag.retriever import ClimateRetriever

            self._climate_retriever = ClimateRetriever()
        return self._climate_retriever

    async def analysis_node(self, state: AgentState) -> AgentState:
        logger.info("Photo Analysis with SigLIP Guidance")
        if not state.get("photo_path"):
            logger.warning("Нет фото")
            return state

        valid_size, size_error = validate_size(state["photo_path"], self.cfg)
        if not valid_size:
            logger.error(size_error)
            state["errors"].append(size_error)
            return state

        valid_type, type_error = validate_type(state["photo_path"], self.cfg)
        if not valid_type:
            logger.error(type_error)
            state["errors"].append(type_error)
            return state

        siglip = self.get_siglip()

        try:
            climate_context = state.get("rag_context", "")

            siglip_prediction = None
            if siglip:
                siglip_prediction = await siglip.predict(
                    state["photo_path"],
                    state.get("lat"),
                    state.get("lon"),
                    state.get("temperature"),
                )
                logger.info(
                    f"SigLIP предсказание: {siglip_prediction['season']} (уверенность: {siglip_prediction['confidence']:.2%})"
                )
                logger.info(
                    f"SigLIP вероятности: {json.dumps(siglip_prediction.get('probabilities', {}))}"
                )

            vision_result = await self.analyze_photo(
                self.cfg,
                state["photo_path"],
                state.get("lat"),
                state.get("lon"),
                state.get("city"),
                state.get("temperature"),
                climate_context,
                siglip_prediction=siglip_prediction,
            )

            vision_analysis = parse(vision_result)
            logger.info("ПРЕДСКАЗАНИЯ МОДЕЛЕЙ:")
            logger.info(
                f"   SigLIP: {siglip_prediction['season'] if siglip_prediction else 'N/A'} (вес 70%)"
            )
            logger.info(
                f"   Vision: {vision_analysis.get('season', 'unknown')} (вес 30%)"
            )
            logger.info(
                f"   Финальный сезон: {vision_analysis.get('season', 'unknown')}"
            )

            state["photo_raw_response"] = vision_result
            state["photo_analysis"] = {
                "season": vision_analysis.get("season", "unknown"),
                "month": vision_analysis.get("month", "unknown"),
                "confidence": vision_analysis.get("confidence", "medium"),
                "vision_season": vision_analysis.get("season", "unknown"),
                "vision_confidence": vision_analysis.get("confidence", "medium"),
                "vision_season_probabilities": vision_analysis.get(
                    "season_probabilities", {}
                ),
                "vision_month_probabilities": vision_analysis.get(
                    "month_probabilities", {}
                ),
                "siglip_season": siglip_prediction["season"]
                if siglip_prediction
                else None,
                "siglip_confidence": siglip_prediction["confidence"]
                if siglip_prediction
                else None,
                "siglip_probabilities": siglip_prediction.get("probabilities", {})
                if siglip_prediction
                else {},
                "method": "Vision + SigLIP (SigLIP priority 70%)",
            }

            state["messages"].append(
                {
                    "role": "assistant",
                    "content": f" Анализ: сезон={vision_analysis.get('season', 'unknown')}, месяц={vision_analysis.get('month', 'unknown')}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "photo_analysis",
                }
            )

        except Exception as e:
            logger.error(f"Ошибка в analysis_node: {e}")
            state["errors"].append(str(e))
            state["photo_analysis"] = {
                "season": "unknown",
                "month": "unknown",
                "confidence": "low",
            }

        return state

    async def climate_node(self, state: AgentState) -> AgentState:
        logger.info("Climate node")
        logger.info(
            f"   lat: {state.get('lat')}, lon: {state.get('lon')}, city: {state.get('city')}, temperature: {state.get('temperature')}"
        )

        if state.get("rag_context"):
            return state

        context_parts = []
        logger.info("Retriever")
        retriever = self.get_retriever()

        if state.get("city"):
            logger.info("Retriever city")
            rag_context = retriever.get_climate_context(city=state["city"])
            if rag_context:
                context_parts.append(f"LOCAL CLIMATE KNOWLEDGE:\n{rag_context}")
                logger.info(f"RAG data found for city: {state['city']}")
            else:
                logger.info(f"No RAG data for city: {state['city']}")
        elif state.get("lat") and state.get("lon"):
            rag_context = retriever.get_climate_context(
                lat=state["lat"], lon=state["lon"]
            )
            if rag_context:
                context_parts.append(f"LOCAL CLIMATE KNOWLEDGE:\n{rag_context}")
                logger.info("RAG data found for coordinates")

        if context_parts:
            state["rag_context"] = "\n\n---\n\n".join(context_parts)
        else:
            state["rag_context"] = "No climate data available for this location."

        return state

    async def synthesis_node(self, state: AgentState) -> AgentState:
        logger.info(" Synthesis node")

        photo = state.get("photo_analysis", {})
        climate = state.get("rag_context", "")
        user_message = state.get("user_message", "")

        state["synthesized"] = {
            "season": photo.get("season", "unknown"),
            "month": photo.get("month", "unknown"),
            "confidence": photo.get("confidence", "medium"),
        }
        siglip_info = ""
        if photo.get("siglip_season"):
            siglip_info = f"""
SigLIP анализ:
- Предсказанный сезон: {photo["siglip_season"]}
- Уверенность: {photo["siglip_confidence"]:.2%}
- Вероятности: {json.dumps(photo.get("siglip_probabilities", {}), ensure_ascii=False)}
"""

        prompt = f"""
Ты помощник, который определяет сезон по фотографии.

Данные с фотографии (ансамбль Vision + SigLIP):
- Сезон: {photo.get("season", "unknown")}
- Месяц: {photo.get("month", "unknown")}
- Уверенность: {photo.get("confidence", "medium")}

{siglip_info}

Климатический контекст:
{climate}

Вопрос пользователя: {user_message}

SigLIP модель обучена на реальных климатических данных (температура + координаты) и имеет приоритет.
Ответь полезно и дружелюбно.
"""

        logger.info(f"DEBUG KEY: {str(GROQ_API_KEY)[:6]}...{str(GROQ_API_KEY)[-4:]}")

        response = await groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )

        state["answer"] = response.choices[0].message.content
        logger.info("Ответ сгенерирован")

        return state

    def formatter_node(self, state: AgentState) -> AgentState:
        logger.info("Formatter Node")

        photo = state.get("photo_analysis", {})

        season_ru_map = self.cfg.graph.SEASON_NAMES_RU
        month_ru_map = self.cfg.graph.MONTH_NAMES_RU

        siglip_probs = photo.get("siglip_probabilities", {})
        vision_probs = photo.get("vision_season_probabilities", {})

        combined_season = {}
        if siglip_probs and vision_probs:
            for s in ["winter", "spring", "summer", "autumn"]:
                combined_season[s] = (
                    siglip_probs.get(s, 0) * 0.7 + vision_probs.get(s, 0) * 0.3
                )
        elif siglip_probs:
            combined_season = siglip_probs
        elif vision_probs:
            combined_season = vision_probs
        else:
            final_season = photo.get("season", "unknown")
            if final_season != "unknown":
                combined_season = {final_season: 1.0}

        total = sum(combined_season.values())
        if total > 0:
            combined_season = {k: v / total for k, v in combined_season.items()}

        sorted_seasons = sorted(
            combined_season.items(), key=lambda x: x[1], reverse=True
        )

        month_probs = photo.get("vision_month_probabilities", {})
        sorted_months = (
            sorted(month_probs.items(), key=lambda x: x[1], reverse=True)
            if month_probs
            else []
        )

        confidence = photo.get("confidence", "medium")
        if isinstance(confidence, float):
            if confidence > 0.8:
                conf_label = "ВЫСОКАЯ"
            elif confidence > 0.6:
                conf_label = "СРЕДНЯЯ"
            else:
                conf_label = "НИЗКАЯ"
        else:
            conf_label = {"high": "ВЫСОКАЯ", "medium": "СРЕДНЯЯ", "low": "НИЗКАЯ"}.get(
                confidence, "СРЕДНЯЯ"
            )

        lines = []

        if photo.get("siglip_season"):
            siglip_ru = season_ru_map.get(
                photo["siglip_season"], photo["siglip_season"]
            )
            siglip_pct = int(round(photo.get("siglip_confidence", 0) * 100))
            lines.append(f"SigLIP Предсказание: {siglip_ru} ({siglip_pct}%)")

        lines.append("РЕЗУЛЬТАТЫ АНАЛИЗА")
        lines.append("")

        for season, prob in sorted_seasons:
            ru_name = season_ru_map.get(season, season)
            pct = round(prob * 100, 1)
            lines.append(f"  {ru_name}: {pct}%")

        if sorted_months:
            lines.append("")
            top_months = [(m, p) for m, p in sorted_months if p > 0][:5]
            for month_en, prob in top_months:
                ru_name = month_ru_map.get(month_en, month_en)
                pct = round(prob * 100, 1)
                lines.append(f"  {ru_name}: {pct}%")

        lines.append(f"Уверенность: {conf_label}")

        state["answer"] = "\n".join(lines)
        return state


print("Close", flush=True)
print("EE")
