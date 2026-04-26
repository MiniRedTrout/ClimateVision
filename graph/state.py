from typing import TypedDict,Optional,List,Dict,Any 
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, field
@dataclass 
class AgentState:
    """Состояние"""
    user_id: Optional[int]
    user_message:Optional[str] = ""
    photo_path:Optional[str] = None 
    lat:Optional[float] = None 
    lon: Optional[float] = None 
    city: Optional[str] = None 
    route: Optional[str] = None 
    has_photo: bool = False 
    has_location: bool = False 
    photo_analysis: Optional[Dict[str,Any]] = None 
    photo_raw_response: Optional[str] = None 
    rag_context: Optional[str] = None 
    last_llm_response: Optional[str] = None 
    synthesized: Optional[Dict[str, Any]] = None 
    tool_result: Dict[str,List[str]] = field(default_factory=list)
    answer: Optional[str] = None 
    errors: List[str] = field(default_factory=list)
    messages: List[BaseMessage] = field(default_factory=list)


