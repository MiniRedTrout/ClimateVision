# graph/__init__.py
# Уберите импорты из tools, оставьте только builder
from .builder import build_agent_graph
from .state import AgentState

# Если нужно экспортировать tools, делайте это лениво
def __getattr__(name):
    if name in ['get_climate_history', 'get_seasonal_forecast', 'get_climate_normals', 'VISION_TOOLS']:
        from .tools import get_climate_history, get_seasonal_forecast, get_climate_normals, VISION_TOOLS
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")