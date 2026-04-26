from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import AgentNodes
from .tools import ALL_TOOLS
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage

def build_agent_graph(cfg, ollama_client, climate_retriever, analyze_photo_func):
    workflow = StateGraph(AgentState)
    nodes = AgentNodes(cfg, ollama_client, climate_retriever, analyze_photo_func)
    tool_node = ToolNode(ALL_TOOLS)
    def should_continue(state: AgentState):
        last_message = state.get('last_llm_response')
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "finalize"
    workflow.add_node("router", nodes.router_node)
    workflow.add_node("tools", tool_node)   
    workflow.add_node("climate", nodes.climate_node)   
    workflow.add_node("analysis", nodes.analysis_node)
    workflow.add_node("synthesis", nodes.synthesis_node)
    workflow.add_node("formatter", nodes.formatter_node)
    workflow.set_entry_point("router")
    workflow.add_edge("router", 'climate')
    workflow.add_edge("climate", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_conditional_edges(
        "synthesis",
        should_continue,
        {
            "tools": "tools",
            "formatter": "formatter"
        }
    )
    workflow.add_edge("tools", "synthesis")
    workflow.add_edge("formatter", END)
    return workflow.compile()