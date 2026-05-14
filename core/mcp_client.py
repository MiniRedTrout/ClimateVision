# core/mcp_client.py
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
    
    async def start_server(self):
        server_path = Path(__file__).parent / "mcp_servers/openmeteo_server.py"
        if not server_path.exists():
            logger.error(f'MCP server not found at {server_path}')
            return False 
        
        try:
            logger.info(f"Starting MCP server from {server_path}")
            
            # Запускаем сервер как subprocess
            self.process = await asyncio.create_subprocess_exec(
                'python', '-u', str(server_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Даем серверу время на инициализацию
            await asyncio.sleep(2)
            
            # Проверяем, не упал ли процесс
            if self.process.returncode is not None:
                stderr = await self.process.stderr.read()
                logger.error(f"MCP server failed: {stderr.decode()}")
                return False
            
            logger.info("MCP server started successfully")
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
            logger.warning(f"MCP server died, restarting...")
            self.server_ready = False
            return await self.ensure_connection()
        
        return True
    
    async def call_tool(self, tool_name: str, timeout: int = 60, **kwargs) -> str:
        logger.info(f"Calling MCP tool: {tool_name} with args: {kwargs}")
        
        await self.ensure_connection()
        
        # JSON-RPC запрос
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
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()
            
            # Читаем ответ
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout
            )
            
            if not response_line:
                raise ConnectionError("Empty response from MCP server")
            
            response = json.loads(response_line.decode())
            
            if "result" in response:
                content = response["result"].get("content", [])
                if content:
                    return content[0].get("text", "")
                return json.dumps({"status": "success"})
            elif "error" in response:
                return json.dumps({"status": "error", "message": response["error"].get("message")})
            else:
                return json.dumps({"status": "error", "message": "Unknown response"})
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling {tool_name}")
            self.server_ready = False
            return json.dumps({"status": "error", "message": "Timeout"})
        except Exception as e:
            logger.error(f"MCP call error: {e}")
            self.server_ready = False
            return json.dumps({"status": "error", "message": str(e)})
    
    async def get_climate_history(self, lat: float, lon: float, year: int = 2023) -> str:
        cache_key = f'climate:{lat}:{lon}:{year}'
        cached = climate_cache.get(cache_key)
        if cached:
            return cached 
        result = await self.call_tool("get_climate_history", lat=lat, lon=lon, year=year)
        climate_cache.set(cache_key, result, ttl=86400)
        return result 
    
    async def get_seasonal_forecast(self, lat: float, lon: float) -> str:
        return await self.call_tool("get_seasonal_forecast", lat=lat, lon=lon)
    
    async def get_climate_normals(self, lat: float, lon: float) -> str:
        return await self.call_tool("get_climate_normals", lat=lat, lon=lon)
    
    async def close(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.server_ready = False