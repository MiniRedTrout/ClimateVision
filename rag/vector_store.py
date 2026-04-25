
import os
import json
import numpy as np
from typing import List, Dict, Tuple



class VectorStore:
    def __init__(self, knowledge_path: str):
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.vectors = {}
        self._build_vectors()
    
    def _build_vectors(self):
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]

        for city_key, city_data in self.data.items():
            vector = []
            monthly = city_data.get("monthly", {})
            for month in month_order:
                if month in monthly:
                    vector.append(monthly[month].get("temp", 0))
                else:
                    vector.append(0)
            self.vectors[city_key] = np.array(vector)
    
    def find_similar_by_climate(self, lat: float, lon: float, top_k: int = 3) -> List[Tuple[str, Dict, float]]:
        nearest = None
        min_dist = float('inf')

        for city_key, city_data in self.data.items():
            city_lat = city_data.get("lat")
            city_lon = city_data.get("lon")
            
            if city_lat is None or city_lon is None:
                continue

            dist = np.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = city_key
        if not nearest or nearest not in self.vectors:
            return []
        target_vector = self.vectors[nearest]
        similarities = []
        for city_key, vector in self.vectors.items():
            if city_key == nearest:
                continue
            
            cos_sim = np.dot(target_vector, vector) / (np.linalg.norm(target_vector) * np.linalg.norm(vector) + 1e-8)
            similarities.append((city_key, cos_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for city_key, score in similarities[:top_k]:
            if score > 0.7:
                result.append((city_key, self.data[city_key], score))
        
        return result
    def find_similar_by_vector(self,vector:np.ndarray,top_k: int = 3)->List[Tuple[str,Dict, float]]:
        if not self.vectors:
            return []
        similarities = []
        for city_key,city_vector in self.vectors.items():
            cos_sim = np.dot(vector,city_vector)/(np.linalg.norm(vector) * np.linalg.norm(city_vector) + 1e-8)
            similarities.append((city_key,cos_sim))
        similarities.sort(key=lambda x: x[1],reverse=True)
        result = []
        for city_key, score in similarities[:top_k]:
            if score > 0.7:
                result.append((city_key, self.data[city_key], score))
        return result 