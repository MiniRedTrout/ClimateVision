import os 
import json 
import math 
from typing import Dict, Optional, Tuple
from .vector_store import VectoreStore 
from utils import logger 

class ClimateRetriever:
    """Ищем данные по локальной базе"""
    def __init__(self,path: str=None):
        if path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(curr_dir,'knowledge_base.json')
        if not os.path.exists(path):
            logger.warning(f"Knowledge file not found")
            self.data = {}
        else:
            with open(path,'r',encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f'Loaded {len(self.data)} cities')
        self.vector_store = VectoreStore(path) if self.data else None 
    def find_city_coords(self,lat:float,lon:float)->Tuple[Optional[str],Optional[dict]]:
        nearest_city = None 
        nearest_data = None 
        min_dis = float('inf')
        for city_key,city_data in self.data.items():
            city_lat = city_data.get('lat')
            city_lon = city_data.get('lon')
            if city_lat is None or city_lon is None:
                continue 
            dis = math.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
            if dis < min_dis:
                min_dis = dis
                nearest_city=city_key
                nearest_data =city_data
        if min_dis < 2.0:
            return nearest_city,nearest_data
        return None,None
    def find_city_name(self,city_name:str)->Optional[Dict]:
        city = city_name.lower()
        for city_data in self.data.values():
            if city in city_data.get('city','').lower():
                return city_data 
        return None 
    def get_climate_context(self, lat: float = None, lon: float = None, city: str = None) -> str:
        city_data = None
        if city:
            city_data = self.find_city_by_name(city)
        if not city_data and lat and lon:
            _, city_data = self.find_city_by_coords(lat, lon)
        if not city_data:
            return ""
        return self._format_context(city_data)
    def _format_context(self, city_data: Dict) -> str:
        city_name = city_data.get("city", "Unknown")
        monthly = city_data.get("monthly", {})
        context = f"\nCLIMATE KNOWLEDGE: {city_name}\n"
        context += "=" * 50 + "\n"
        for month in ["December", "January", "February", "March", "April"]:
            if month in monthly:
                m = monthly[month] 
                context += f"{month:10} | {m.get('season', ''):6} | {m.get('temp', 0):5.1f}°C | snow: {m.get('snow', 0):3.0f}mm\n"
        context += "=" * 50 + "\n"
        context += "RULE: Use this climate data as PRIMARY reference. If March has temp > 0°C → it's SPRING.\n"
        return context