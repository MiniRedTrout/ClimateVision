
import asyncio
import subprocess
import json
from pathlib import Path

class OpenMeteoMCPClient:
    
    def __init__(self):
        self.process = None
        self.server_ready = False
    
    async def start_server(self):
        server_path = Path(__file__).parent / "mcp_servers/openmeteo_server.py"
        
        self.process = await asyncio.create_subprocess_exec(
            'python', str(server_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.sleep(2)
        self.server_ready = True
        print(" Open-Meteo MCP Server started")
    
    async def call_tool(self, tool_name: str, **kwargs) -> str:
        if not self.server_ready:
            await self.start_server()

        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            },
            "id": 1
        }
        
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        await self.process.stdin.drain()
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line)
        
        if "result" in response and "content" in response["result"]:
            return response["result"]["content"][0]["text"]
        
        return "Error calling MCP tool"
    
    async def get_current_weather(self, lat: float, lon: float) -> str:
        return await self.call_tool("get_current_weather", lat=lat, lon=lon)
    
    async def get_climate_history(self, lat: float, lon: float, year: int = 2023) -> str:
        return await self.call_tool("get_climate_history", lat=lat, lon=lon, year=year)
    
    async def get_forecast(self, lat: float, lon: float, days: int = 3) -> str:
        return await self.call_tool("get_forecast", lat=lat, lon=lon, days=days)
    
    async def close(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()