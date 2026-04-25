import torch 
import numpy as np 
from PIL import Image 
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm 
import json
import pickle 
import faiss 
from typing import List, Dict, Tuple 

class SigLIPIndexer:
    def __init__(self, dir: str ='reference_images', device: str = 'cpu'):
        self.dir = Path(dir)
        self.device = device
        self.model_name = "google/siglip-base-patch16-256-i18n"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()
        self.index = None 
        self.metadata = []
    def get_embedding(self,path:str)->np.ndarray:
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image,return_tensors='pt').to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        embedding = embedding/embedding.norm(dim=-1,keepdim=True)
        return embedding.cpu().numpy().flatten()
    def build_index(self,batch_size:int = 32):
        image_paths = []
        locations = []
        for img_path in self.dir.rglob("*.jpg"):
            if img_path.parent.name in ["january", "february", "march", "april", "may", "june",
                                         "july", "august", "september", "october", "november", "december"]:
                image_paths.append(img_path)
                season = img_path.parent.parent.name 
                month = img_path.parent.name 
                coord_file = img_path.with_suffix('.jpg.txt')
                lat,lon = None,None 
                if coord_file.exists():
                    with open(coord_file) as f:
                        for line in f:
                            if 'lat:' in line:
                                lat = float(line.split(':')[1].strip())
                            elif 'lon:' in line:
                                lon = float(line.split(':')[1].strip())
                locations.append({
                    "path": str(img_path),
                    "season": season,
                    "month": month,
                    "lat": lat,
                    "lon": lon
                })
        all_embeddings = []
        for i in tqdm(range(0,len(image_paths),batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
            inputs = self.processor(images=batch_images,return_tensors='pt',padding=True).to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings/embeddings.norm(dim=-1,keepdim=True)
            all_embeddings.extend(embeddings.cpu().numpy())
        dimension = all_embeddings[0].shape[0]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(all_embeddings))
        self.metadata = locations 
        faiss.write_index(self.index,'siglip_index.faiss')
        with open('siglip_metadata.pkl','wb') as f:
            pickle.dump(self.metadata,f)
        return self.index 
    def load_index(self,index_path:str="siglip_index.faiss", metadata_path: str = "siglip_metadata.pkl"):
        self.index = faiss.read_index(index_path)
        with open(metadata_path,'rb') as f:
            self.metadata = pickle.load(f)
        return self.index 
    def find_similar(self, path:str, top_k: int = 5)-> List[Dict]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        query_embedding = self.get_embedding(path).reshape(1,-1)
        scores,indices = self.index.search(query_embedding,top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity'] = float(scores[0][i])
                results.append(result)
        return results 


def create_index():
    indexer = SigLIPIndexer(reference_dir="reference_images")
    indexer.build_index()

if __name__ == "__main__":
    create_index()