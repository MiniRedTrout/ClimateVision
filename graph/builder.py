from langgraph.graph import StateGraph, END
print('seek_problem',flush=True)
from .state import AgentState
print('Да что такое', flush=True)
from .nodes import AgentNodes
print('build',flush=True)
def build_agent_graph(cfg, analyze_photo_func):
    workflow = StateGraph(AgentState)
    nodes = AgentNodes(cfg, analyze_photo_func)
    workflow.add_node("router", nodes.router_node)  
    workflow.add_node("climate", nodes.climate_node)   
    workflow.add_node("analysis", nodes.analysis_node)
    workflow.add_node("synthesis", nodes.synthesis_node)
    workflow.add_node("formatter", nodes.formatter_node)
    workflow.set_entry_point("router")
    workflow.add_edge("router", 'climate')
    workflow.add_edge("climate", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis","formatter")
    workflow.add_edge("formatter", END)
    return workflow.compile()