import math 
import json
from typing import Dict, Optional, Tuple 
from .vector_store import VectorStore 

class ClimateRetriever:
    def __init__(self, knowledge_path: str ='rag/knowledge_base.json'):
        self.data = self._load_data(knowledge_path)
        self.vector_store = VectorStore(knowledge_path)
    def _load_data(self,path:str)->Dict:
        with open(path,'r',encoding='utf-8') as f:
            return json.load(f)
    def find_city_by_coords(self, lat:float,lon:float)->Tuple[Optional[str], Optional[Dict]]:
        min_distance = float('inf')
        nearest_city = None
        nearest_data = None
        for city_key, city_data in self.data.items():
            city_lat = city_data["lat"]
            city_lon = city_data["lon"]
            distance = math.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_city = city_key
                nearest_data = city_data

        if min_distance < 2.0:
            return nearest_city, nearest_data
        return None, None
    def get_climate_context(self, lat: float = None, lon: float = None, city: str = None) -> str:
        city_data = None
        city_key = None
        if city:
            for key, data in self.data.items():
                if city.lower() in data["city"].lower():
                    city_key = key
                    city_data = data
                    break

        if not city_data and lat and lon:
            city_key, city_data = self.find_city_by_coords(lat, lon)
        if not city_data and lat and lon:
            similar = self.vector_store.find_similar_by_climate(lat, lon, top_k=1)
            if similar:
                city_key, city_data, score = similar[0]
                print(f"Found similar city by climate: {city_data['city']} (similarity: {score:.2f})")
        
        if not city_data:
            return ""
        
        return self._format_context(city_data)
    
    def _format_context(self, city_data: Dict) -> str:
        city_name = city_data["city"]
        monthly = city_data.get("monthly", {})
        
        context = f"\n🏙️ CLIMATE KNOWLEDGE: {city_name}\n"
        context += "=" * 50 + "\n"
        
        for month in ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]:
            if month in monthly:
                m = monthly[month]
                season_symbol = {"winter": "❄️", "spring": "🌸", "summer": "☀️", "autumn": "🍂"}.get(m["season"], "")
                context += f"{month:10} | {season_symbol} {m['season']:6} | {m['temp']:5.1f}°C | snow: {m['snow']:3.0f}mm\n"
        
        context += "=" * 50 + "\n"
        context += "RULE: Use this climate data as PRIMARY reference. If March has temp > 0°C → it's SPRING.\n"
        
        return context