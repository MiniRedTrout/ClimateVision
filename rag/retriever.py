import os 
import json 
import math 
from typing import Dict, Optional, Tuple,List
from .vector_store import VectorStore 
from utils import logger 
from sentence_transformers import SentenceTransformer
import numpy as np

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
        self.embedding_model = None 
        self.city_names = []
        self.city_embeddings = []
        self.city_descriptions = []
        self._search_index_built = False
        self.vector_store = VectorStore(path) if self.data else None 
    def _build_search_index(self):
        if self._search_index_built:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.city_names = []
            self.city_descriptions = []
            for city_key, city_data in self.data.items():
                city_name = city_data.get('city', city_key)
                self.city_names.append(city_name)
                monthly = city_data.get('monthly', {})
                description = f"{city_name}."
                for month, data in monthly.items():
                    description += f"{month}:{data.get('temp',0)}C, {data.get('season','')}."
                self.city_descriptions.append(description)
            
            if self.city_descriptions:
                self.city_embeddings = self.embedding_model.encode(self.city_descriptions)
            
            self._search_index_built = True
            logger.info(f"Semantic search index built for {len(self.city_names)} cities")
        except Exception as e:
            logger.warning(f'Could not load embedding model: {e}')
            self.embedding_model = None
            self._search_index_built = True 
    def search_by_text(self, query: str, top_k: int = 3)-> List[Tuple[str,Dict,float]]:
        self._ensure_search_index()  
        if not self.embedding_model or not self.city_embeddings:
            return []
        
        import numpy as np
        query_embedding = self.embedding_model.encode([query])
        similarities = np.dot(self.city_embeddings, query_embedding.T).flatten()
        top = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top:
            city_name = self.city_names[idx]
            city_data = self.data.get(city_name, self.data.get(city_name.lower()))
            if city_data:
                results.append((city_name, city_data, float(similarities[idx])))
        return results
    def find_similar_cities(self, city_name:str, top_k:int=3)->List[Tuple[str,Dict,float]]:
        self._ensure_search_index() 
        if not self.embedding_model or not self.city_names:
            return []
        
        import numpy as np
        
        # Поиск города
        if city_name not in self.city_names:
            for name in self.city_names:
                if city_name.lower() in name.lower():
                    city_name = name
                    break
        
        if city_name not in self.city_names:
            return []
        
        idx = self.city_names.index(city_name)
        target_embedding = self.city_embeddings[idx]
        similarities = np.dot(self.city_embeddings, target_embedding)
        top = similarities.argsort()[-top_k-1:][::-1]
        
        results = []
        for i in top:
            if i != idx:
                similar_city = self.city_names[i]
                city_data = self.data.get(similar_city)
                if city_data:
                    results.append((similar_city, city_data, float(similarities[i])))
        return results[:top_k]
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
            city_data = self.find_city_name(city)
        if not city_data and lat and lon:
            _, city_data = self.find_city_coords(lat, lon)
        if not city_data and city and self.embedding_model:
            results = self.search_by_text(city,top_k=1)
            if results:
                _,city_data,score = results[0]
        if not city_data:
            return ""
        return self._format_context(city_data)
    def get_similar_climates_context(self, city_name: str) -> str:
        similar = self.find_similar_cities(city_name, top_k=3)
        if not similar:
            return ""
        context = "\n**SIMILAR CLIMATES:**\n"
        for city, data, score in similar:
            monthly = data.get('monthly', {})
            winter_temps = []
            for month in ["December", "January", "February"]:
                if month in monthly:
                    winter_temps.append(monthly[month].get('temp', 0))
            avg_winter = sum(winter_temps) / len(winter_temps) if winter_temps else 0
            context += f"• **{city}** (similarity: {score:.2f}): {avg_winter:.1f}°C in winter\n"
        return context
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