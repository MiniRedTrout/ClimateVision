from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Состояние агента, передаваемое между узлами LangGraph"""
    user_id: Optional[int]
    user_message: Optional[str]
    photo_path: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    city: Optional[str]
    route: Optional[str]
    has_photo: bool
    has_location: bool
    photo_analysis: Optional[Dict[str, Any]]
    photo_raw_response: Optional[str]
    rag_context: Optional[str]
    last_llm_response: Optional[Any]
    synthesized: Optional[Dict[str, Any]]
    tool_result: List[Dict[str, Any]]
    answer: Optional[str]
    errors: List[str]
    messages: List[BaseMessage]
