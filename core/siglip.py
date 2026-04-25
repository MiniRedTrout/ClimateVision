import torch
from transformers import AutoProcessor, AutoModel 
from PIL import Image 
import numpy as np
from pathlib import Path 
from utils import logger 
from typing import Optional, Dict, List 
from core.siglip_indexer import SigLIPIndexer 

class SigLIP:
    _instance = None 
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False 
        return cls._instance
    def __init__(self):
        if self._initialized:
            return 
        self._initialized = True 
        self.indexer = SigLIPIndexer()
        index_path = Path('siglip_index.faiss')
        metadata_path = Path('siglip_metadata.pkl')
        if index_path.exists() and metadata_path.exists():
            self.indexer.load_index()
            self.enabled = True 
        else:
            self.enabled = False 
    def find_similar(self, path: str, lat:Optional[float]=None, lon:Optional[float] = None)->Dict:
        if not self.enabled:
            return {}
        similar = self.indexer.find_similar(path,top_k=10)
        if not similar:
            return {}
        if lat and lon:
            for item in similar:
                if item.get('lat') and item.get('lon'):
                    import math 
                    dist = math.hypot(lat - item['lat'],lon - item['lon'])
                    similarity_weight = 1 - min(1,dist/10)
                    item['weighted_similarity'] = item['similarity'] * (0.7 + 0.3 * similarity_weight)
                    similar.sort(key=lambda x: x.get('weighted_similarity',0),reverse=True)
                    best = similar[0]
                    return {
                        "season": best['season'],
                        "month": best['month'],
                        "similarity": best.get('similarity', 0),
                        "source": "siglip"
                    }

    
siglip = SigLIP()