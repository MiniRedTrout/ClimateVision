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
        self.reader = None
        self.writer = None 
    
    async def start_server(self):
        server_path = Path(__file__).parent / "mcp_servers/openmeteo_server.py"
        if not server_path.exists():
            logger.error(f'MCP server not found at {server_path}')
            return False 
        
        try:
            logger.info(f"Starting MCP server from {server_path}")
            self.process = await asyncio.create_subprocess_exec(
                'python', '-u', str(server_path),  # -u для unbuffered output
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Даем серверу время на запуск
            await asyncio.sleep(3)
            
            # Проверяем, жив ли процесс
            if self.process.returncode is not None:
                stderr = await self.process.stderr.read()
                logger.error(f"MCP server failed to start: {stderr.decode()}")
                return False
            
            logger.info("Open-Meteo MCP Server started successfully")
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
            logger.warning(f"MCP server died with code {self.process.returncode}")
            # Читаем stderr для диагностики
            if self.process.stderr:
                stderr = await self.process.stderr.read()
                logger.error(f"MCP stderr: {stderr.decode()}")
            self.server_ready = False
            return await self.ensure_connection()
        
        return True
    
    async def call_tool(self, tool_name: str, timeout: int = 60, **kwargs) -> str:
        logger.info(f"Calling MCP tool: {tool_name} with args: {kwargs}")
        
        await self.ensure_connection()
        
        # Отправляем запрос в формате JSON-RPC
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
            logger.debug(f"Sending: {request_json.strip()}")
            
            self.process.stdin.write(request_json.encode())
            await asyncio.wait_for(self.process.stdin.drain(), timeout=10)
            
            # Читаем ответ построчно
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(), 
                timeout=timeout
            )
            
            if not response_line:
                logger.error("Empty response from MCP server")
                # Проверяем статус процесса
                if self.process.returncode is not None:
                    logger.error(f"Process died with code: {self.process.returncode}")
                    if self.process.stderr:
                        stderr = await self.process.stderr.read()
                        logger.error(f"Stderr: {stderr.decode()}")
                raise ConnectionError("Empty response from MCP server")
            
            response = json.loads(response_line.decode())
            logger.debug(f"Received: {response}")
            
            # Обрабатываем ответ
            if "result" in response:
                content = response["result"].get("content", [])
                if content and len(content) > 0:
                    return content[0].get("text", "")
                return json.dumps({"status": "success", "message": "No content"})
            elif "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"MCP error: {error_msg}")
                return json.dumps({"status": "error", "message": error_msg})
            else:
                return json.dumps({"status": "error", "message": "Unexpected response"})
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling MCP tool {tool_name} after {timeout}s")
            self.server_ready = False  
            return json.dumps({"status": "error", "message": "Request timeout"})
        except Exception as e:
            logger.error(f"MCP call error: {e}")
            self.server_ready = False
            return json.dumps({"status": "error", "message": str(e)})
    
    async def get_current_weather(self, lat: float, lon: float) -> str:
        return await self.call_tool("get_current_weather", lat=lat, lon=lon)
    
    async def get_climate_history(self, lat: float, lon: float, year: int = 2023) -> str:
        cache_key = f'climate:{lat}:{lon}:{year}'
        cached = climate_cache.get(cache_key)
        if cached:
            return cached 
        result = await self.call_tool("get_climate_history", lat=lat, lon=lon, year=year)
        climate_cache.set(cache_key, result, ttl=86400)
        return result 
    
    async def get_forecast(self, lat: float, lon: float, days: int = 3) -> str:
        return await self.call_tool("get_forecast", lat=lat, lon=lon, days=days)
    
    async def get_seasonal_forecast(self, lat: float, lon: float) -> str:
        return await self.call_tool("get_seasonal_forecast", lat=lat, lon=lon)
    
    async def get_climate_normals(self, lat: float, lon: float) -> str:
        return await self.call_tool("get_climate_normals", lat=lat, lon=lon)
    
    async def close(self):
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
            self.server_ready = False
            logger.info("MCP server closed")