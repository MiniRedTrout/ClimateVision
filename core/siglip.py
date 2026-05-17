
import aiohttp
import base64
import asyncio
from typing import Optional

class SigLIPClassifier:
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.api_url = "https://miniredtrout-season-model-api.hf.space/predict"
        self.timeout = aiohttp.ClientTimeout(total=60)
    
    async def predict(self, image_path: str, lat: Optional[float] = None, 
                      lon: Optional[float] = None, temperature: Optional[float] = None) -> dict:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        payload = {
            "image": image_base64,
            "lat": lat,
            "lon": lon,
            "temperature": temperature
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'season': result['season'],
                        'confidence': result['confidence'],
                        'probabilities': result.get('probabilities', {})
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")