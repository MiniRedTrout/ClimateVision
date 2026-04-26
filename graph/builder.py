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
    llm_with_tools = ollama_client.bind_tools(ALL_TOOLS)
    def should_continue(state: AgentState):
        messages = state.get('messages', [])
        if not messages:
            return "finalize"
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        if state.get('has_photo') and not state.get('photo_analysis'):
            return "analysis"
        return "finalize"
    workflow.add_node("router", nodes.router_node)
    workflow.add_node("tools", tool_node)      
    workflow.add_node("analysis", nodes.analysis_node)
    workflow.add_node("synthesis", nodes.synthesis_node)
    workflow.add_node("formatter", nodes.formatter_node)
    workflow.set_entry_point("router")
    workflow.add_edge("router", should_continue, {
        "tools": "tools",
        "analysis": "analysis",
        "finalize": "formatter"
    })
    workflow.add_edge("tools", "router")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis", "formatter")
    workflow.add_edge("formatter", END)
    return workflow.compile()