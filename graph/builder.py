
print("=== LOADING graph/builder.py ===", flush=True)

try:
    from langgraph.graph import StateGraph, END
    print("  langgraph.graph imported", flush=True)
except Exception as e:
    print(f"  !!! Error importing langgraph: {e}", flush=True)
    raise
try:
    from .tools import ALL_TOOLS
    print("  ALL TOOLS imported", flush=True)
except Exception as e:
    print(f"  !!! Error importing tools: {e}", flush=True)
    raise

try:
    from langgraph.prebuilt import ToolNode
    print("  Toolode imported", flush=True)
except Exception as e:
    print(f"  !!! Error importing toolnode: {e}", flush=True)
    raise


try:
    from .state import AgentState
    print("  .state imported", flush=True)
except Exception as e:
    print(f"  !!! Error importing AgentState: {e}", flush=True)
    raise

try:
    from .nodes import AgentNodes
    print("  .nodes imported", flush=True)
except Exception as e:
    print(f"  !!! Error importing AgentNodes: {e}", flush=True)
    raise

print("=== graph/builder.py loaded successfully ===", flush=True)


def build_agent_graph(cfg, ollama_client, analyze_photo_func):
    print("=== build_agent_graph called ===", flush=True)
    workflow = StateGraph(AgentState)
    nodes = AgentNodes(cfg, ollama_client, analyze_photo_func)
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