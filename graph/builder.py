from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import AgentNodes

def build_agent_graph(cfg, ollama_client, climate_retriever, analyze_photo_func):
    workflow = StateGraph(AgentState)
    nodes = AgentNodes(cfg, ollama_client, climate_retriever, analyze_photo_func)
    workflow.add_node("router", nodes.router_node)
    workflow.add_node("siglip", nodes.siglip_node) 
    workflow.add_node("analysis", nodes.analysis_node)
    workflow.add_node("synthesis", nodes.synthesis_node)
    workflow.add_node("formatter", nodes.formatter_node)
    workflow.set_entry_point("router")
    workflow.add_edge("router", "siglip")
    workflow.add_edge("siglip", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis", "formatter")
    workflow.add_edge("formatter", END)
    return workflow.compile()