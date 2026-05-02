
import asyncio
import subprocess
import json
from pathlib import Path
from cache import climate_cache
from utils import logger 

class OpenMeteoMCPClient:
    
    def __init__(self):
        self.process = None
        self.server_ready = False
        self.reader = None
        self.writer = None 
    
    async def start_server(self):
        server_path = Path(__file__).parent / "mcp_servers/openmeteo_server.py"
        if not server_path.exists():
            logger.error(f'MCP server not found')
            return False 
        try:
          self.process = await asyncio.create_subprocess_exec(
            'python', str(server_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
          )
          await asyncio.sleep(2)
          if self.process.returncode is not None:
                stderr = await self.process.stderr.read()
                logger.error(f"MCP server failed to start: {stderr.decode()}")
                return False
          
          self.server_ready = True
          return True 
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    async def ensure_connection(self):
        if not self.server_ready:
            success = await self.start_server()
            if not success:
                raise ConnectionError("Cannot connect to MCP server")
        if self.process.returncode is not None:
            logger.warning("MCP server died, restarting...")
            self.server_ready = False
            return await self.ensure_connection()
        
        return True
    async def call_tool(self, tool_name: str, **kwargs) -> str:

        await self.ensure_connection()

        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            },
            "id": 1
        }
        try:
            # Отправляем запрос
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode())
            await asyncio.wait_for(self.process.stdin.drain(), timeout=5)
            
            # Читаем ответ с таймаутом
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(), 
                timeout=timeout
            )
            
            if not response_line:
                raise ConnectionError("Empty response from MCP server")
            
            response = json.loads(response_line.decode())
            
            if "result" in response and "content" in response["result"]:
                return response["result"]["content"][0]["text"]
            elif "error" in response:
                logger.error(f"MCP error: {response['error']}")
                return f"Error: {response['error'].get('message', 'Unknown error')}"
            else:
                return "Unexpected response from MCP server"
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling MCP tool {tool_name}")
            self.server_ready = False  # Сброс состояния для переподключения
            return "Error: Request timeout. Please try again."
        except Exception as e:
            logger.error(f"MCP call error: {e}")
            self.server_ready = False
            return f"Error: {str(e)}"
    
    async def get_current_weather(self, lat: float, lon: float) -> str:
        return await self.call_tool("get_current_weather", lat=lat, lon=lon)
    
    async def get_climate_history(self, lat: float, lon: float, year: int = 2023) -> str:
        cache_key = f'climate:{lat}:{lon}:{year}'
        cached = climate_cache.get(cache_key)
        if cached:
            return cached 
        result = await self.call_tool("get_climate_history", lat=lat, lon=lon, year=year)
        climate_cache.set(cache_key,result,ttl=86400)
        return result 
    
    async def get_forecast(self, lat: float, lon: float, days: int = 3) -> str:
        return await self.call_tool("get_forecast", lat=lat, lon=lon, days=days)
    
    async def close(self):
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
            self.server_ready = False
            logger.info("MCP server closed")