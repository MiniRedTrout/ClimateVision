import json 
import numpy as np
from typing import List, Dict, Tuple, Optional 

class VectorStore:
    def __init__(self, knowledge_path:str = 'rag/knowledge_base.json'):
        self.knowledge_path = knowledge_path
        self.data = self._load_data()
        self.vectors = {}
        self._build_vectors()
    def _load_data(self) -> Dict:
        with open(self.knowledge_path,'r',encoding='utf-8') as f:
            return json.load(f)
    def _build_vectors(self):
        for city_key,city_data in self.data.items():
            vector = []
            for month in ["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"]:
                if month in city_data.get('monthly',{}):
                    temp = city_data['monthly'][month]['temp']
                    vector.append(temp)
                else:
                    vector.append(0)
            self.vectors[city_key] = np.array(vector)
    def find_similar_cities(self, lat: float, lon:float,top_k: int=3) -> List[Tuple[str, Dict,float]]:
        distances = []
        for city_key, city_data in self.data.items():
            city_lat = city_data['lat']
            city_lon = city_data['lon']
            distance = np.sqrt((lat-city_lat)**2+(lon-city_lon)**2)
            distances.append((city_key,city_data,distance))
        distances.sort(key=lambda x:x[2])
        result = []
        for city_key, city_data,dist in distances[:top_k]:
            if dist < 2.0:
                result.append((city_key,city_data,dist))
        return result 
    def get_city_by_name(self,city_name:str)->Optional[Dict]:
        city_name_lower = city_name.lower()
        for city_key,city_data in self.data.items():
            if city_name_lower in city_data['city'].lower():
                return city_data
        return None 
