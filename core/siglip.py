import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
from pathlib import Path
import os
from utils.logger import logger

class SigLIPClassifier:
    _instance = None 
    @classmethod
    def get_instance(cls,model_path:str=None):
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance
    def __init__(self, model_path: str =None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "siglip-season-model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена в {self.model_path}")
        self.processor = AutoImageProcessor.from_pretrained(str(model_path))
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.seasons = ['winter','spring','summer','autumn']
        self.lat_min, self.lat_max = -90,90
        self.lon_min, self.lon_max = -180,180
        self.temp_min, self.temp_max = -30, 50
        logger.info(f" SigLIP модель загружена")
    def _load_model(self, model_path):
        from transformers import AutoModel
        class Model(nn.Module):
            def __init__(self,model_name,metadata_dim=3, num_classes=4):
                super().__init__()
                self.vision_model = AutoModel.from_pretrained(model_name).vision_model
                self.metadata_encoder = nn.Sequential(
                    nn.Linear(metadata_dim, 64),
                    nn.GELU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64,128),
                    nn.GELU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128,256),
                    nn.GELU()
                )
                hidden_size = self.vision_model.config.hidden_size
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_size + 256,512),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Dropout(0.2)
                )
                self.classifier = nn.Linear(256,num_classes)
            def forward(self,pixel_values, metadata):
                vision_out = self.vision_model(pixel_values)
                vision_features = vision_out.pooler_output
                metadata_features = self.metadata_encoder(metadata)
                combined = torch.cat([vision_features, metadata_features], dim=1)
                fused = self.fusion(combined)
                return self.classifier(fused)
        model = Model("google/siglip-base-patch16-224")
        checkpoint = torch.load(os.path.join(model_path, "best_model.pth"), 
                               map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    def _normalize_metadata(self, lat,lon,temp):
        lat_norm = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1 if lat else 0
        lon_norm = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1 if lon else 0
        temp_norm = 2 * (temp - self.temp_min) / (self.temp_max - self.temp_min) - 1 if temp else 15.0
        return torch.tensor([[lat_norm, lon_norm, temp_norm]], dtype=torch.float32)
    async def predict(self, image_path: str, lat: float = None, lon: float = None, 
                      temperature: float = None) -> dict:
        import asyncio
        
        def _predict():
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            metadata = self._normalize_metadata(lat, lon, temperature).to(self.device)
            
            with torch.no_grad():
                logits = self.model(pixel_values, metadata)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
            
            return {
                'season': self.seasons[pred_idx],
                'confidence': probs[0][pred_idx].item(),
                'probabilities': {self.seasons[i]: probs[0][i].item() for i in range(4)}
            }
        
        return await asyncio.get_event_loop().run_in_executor(None, _predict)
