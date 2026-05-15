# validate_model.py
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Настраиваем пути и окружение ДО всех импортов
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 2. Загружаем .env
env_paths = [
    ROOT_DIR / ".env",
    ROOT_DIR / "config" / ".env",
    Path.cwd() / ".env",
]

for env_path in env_paths:
    if env_path.exists():
        print(f"✅ Загружаем .env из {env_path}")
        load_dotenv(env_path)
        break
else:
    print("⚠️ .env файл не найден")

# 3. Устанавливаем переменные окружения если их нет
if not os.getenv("API_KEY"):
    os.environ["API_KEY"] = "your-groq-api-key-here"
    print("⚠️ API_KEY не найден, установлен заглушкой (замените на реальный ключ)")

if not os.getenv("GROQ_API_BASE"):
    os.environ["GROQ_API_BASE"] = "https://api.groq.com/openai/v1"

# 4. Теперь импортируем все остальное
import asyncio
import json
import tempfile
import shutil
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import hydra
from omegaconf import DictConfig, OmegaConf

# Импорты из проекта
from graph.builder import build_agent_graph
from graph.state import AgentState
from core.analyzer import analyze_photo


class ModelValidator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        print("🚀 Building agent graph...")
        self.agent = build_agent_graph(cfg, analyze_photo)
        self.results = []
        
    async def validate_single_image(self, image_path: str, true_season: str, 
                                     lat: float = None, lon: float = None,
                                     city: str = None) -> Dict:
        """Валидация одного изображения"""
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            shutil.copy2(image_path, tmp.name)
            tmp_path = tmp.name
        
        try:
            initial_state = AgentState(
                user_id=999,
                user_message="Определи сезон по фото",
                photo_path=tmp_path,
                lat=lat,
                lon=lon,
                city=city,
                temperature=None,
                has_photo=True,
                has_location=bool(lat or city),
                route=None,
                photo_analysis=None,
                photo_raw_response=None,
                rag_context=None,
                calendar_context=None,
                synthesized=None,
                answer=None,
                errors=[],
                messages=[]
            )
            
            final_state = await self.agent.ainvoke(initial_state)
            
            # Извлекаем предсказанный сезон
            predicted = "unknown"
            if final_state.get("photo_analysis"):
                predicted = final_state["photo_analysis"].get("season", "unknown")
            elif final_state.get("answer"):
                answer = final_state["answer"].lower()
                for season in ["winter", "spring", "summer", "autumn"]:
                    if season in answer:
                        predicted = season
                        break
            
            return {
                "image_path": image_path,
                "true_season": true_season,
                "predicted_season": predicted,
                "correct": predicted == true_season,
                "error": final_state.get("errors", [None])[0] if final_state.get("errors") else None
            }
            
        except Exception as e:
            return {
                "image_path": image_path,
                "true_season": true_season,
                "predicted_season": "error",
                "correct": False,
                "error": str(e)
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def validate_dataset(self, data_path: str, max_samples: int = None,
                               season_filter: List[str] = None) -> Dict:
        """Валидация датасета"""
        
        print(f"📊 Загрузка данных из {data_path}")
        
        # Загрузка данных
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Фильтрация
        if season_filter:
            data = [item for item in data if item.get('season') in season_filter]
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"📸 Всего образцов: {len(data)}")
        
        # Валидация
        results = []
        for item in tqdm(data, desc="Валидация"):
            # Получаем путь к изображению
            image_path = item.get('image') or item.get('image_path')
            if not image_path:
                print(f"⚠️ Нет пути к изображению в элементе: {item}")
                continue
            
            # Проверяем существование файла
            if not os.path.exists(image_path):
                # Пробуем альтернативные пути
                alt_paths = [
                    ROOT_DIR / "data" / Path(image_path).name,
                    ROOT_DIR / "reference_images" / item.get('season', '') / Path(image_path).name,
                ]
                found = False
                for alt_path in alt_paths:
                    if alt_path.exists():
                        image_path = str(alt_path)
                        found = True
                        break
                
                if not found:
                    print(f"⚠️ Изображение не найдено: {image_path}")
                    continue
            
            result = await self.validate_single_image(
                image_path=image_path,
                true_season=item.get('season', 'unknown'),
                lat=item.get('lat'),
                lon=item.get('lon'),
                city=item.get('city')
            )
            results.append(result)
            
            # Сохраняем промежуточные результаты
            if len(results) % 50 == 0:
                self._save_results(results, "validation_temp.json")
        
        self.results = results
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """Расчет метрик"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        df_valid = df[df['predicted_season'] != 'error']
        
        if len(df_valid) == 0:
            print("⚠️ Нет валидных результатов для расчета метрик")
            return {}
        
        y_true = df_valid['true_season'].tolist()
        y_pred = df_valid['predicted_season'].tolist()
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification Report
        labels = ['winter', 'spring', 'summer', 'autumn']
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Per-class metrics
        class_metrics = {}
        for i, season in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            class_metrics[season] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'support': int(cm[i, :].sum())
            }
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_metrics': class_metrics,
            'total_samples': len(df),
            'valid_samples': len(df_valid),
            'errors': len(df[df['predicted_season'] == 'error'])
        }
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Сохранение результатов"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Промежуточные результаты сохранены в {output_path}")
    
    def print_report(self):
        """Вывод отчета"""
        if not self.results:
            print("⚠️ Нет результатов для отображения")
            return
        
        metrics = self._calculate_metrics()
        
        print("\n" + "="*60)
        print("📊 ОТЧЕТ ПО ВАЛИДАЦИИ МОДЕЛИ")
        print("="*60)
        
        print(f"\n✅ Общая точность (Accuracy): {metrics['accuracy']:.2%}")
        print(f"📸 Всего образцов: {metrics['total_samples']}")
        print(f"✅ Успешных: {metrics['valid_samples']}")
        print(f"❌ Ошибок: {metrics['errors']}")
        
        print("\n📈 Метрики по классам:")
        print("-" * 50)
        for season, m in metrics['class_metrics'].items():
            print(f"  {season.capitalize():8}: "
                  f"Precision={m['precision']:.2%}, "
                  f"Recall={m['recall']:.2%}, "
                  f"F1={m['f1']:.2%}, "
                  f"Support={m['support']}")
        
        print("\n🔄 Матрица ошибок (Confusion Matrix):")
        print("-" * 50)
        labels = ['Winter', 'Spring', 'Summer', 'Autumn']
        print(f"{'':12}", end='')
        for l in labels:
            print(f"{l:>10}", end='')
        print()
        
        cm = metrics['confusion_matrix']
        for i, label in enumerate(labels):
            print(f"{label:12}", end='')
            for j in range(4):
                print(f"{cm[i][j]:>10}", end='')
            print()
        
        # Статистика ошибок
        df = pd.DataFrame(self.results)
        correct = df[df['correct'] == True]
        wrong = df[df['correct'] == False]
        
        print(f"\n📊 Статистика:")
        print(f"  Правильных: {len(correct)} ({len(correct)/len(df)*100:.1f}%)")
        print(f"  Неправильных: {len(wrong)} ({len(wrong)/len(df)*100:.1f}%)")
        
        if len(wrong) > 0:
            print("\n❌ Примеры ошибок (первые 10):")
            for _, row in wrong.head(10).iterrows():
                print(f"  {Path(row['image_path']).name}: "
                      f"true={row['true_season']} -> pred={row['predicted_season']}")


def create_test_dataset_from_csv(csv_path: str, image_root: str, output_json: str):
    """Создание тестового датасета из CSV"""
    df = pd.read_csv(csv_path)
    
    # Определяем колонку с изображениями
    image_col = None
    for col in ['image', 'image_path', 'path', 'file', 'filename']:
        if col in df.columns:
            image_col = col
            break
    
    if not image_col:
        raise ValueError(f"Не найдена колонка с путями к изображениям. Доступные колонки: {df.columns.tolist()}")
    
    data = []
    for _, row in df.iterrows():
        item = {
            'image': os.path.join(image_root, row[image_col]),
            'season': row.get('season', 'unknown'),
            'lat': row.get('lat') if 'lat' in df.columns else None,
            'lon': row.get('lon') if 'lon' in df.columns else None,
            'city': row.get('city') if 'city' in df.columns else None
        }
        data.append(item)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Создан датасет: {len(data)} образцов")
    return output_json


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Запуск валидации модели...")
    
    # Пути к вашему датасету
    DATASET_ROOT = Path("D:/Climate/climate_dataset/processed")
    
    VALIDATION_CONFIG = {
        # Ваш JSON файл с данными
        "data_path": str(DATASET_ROOT / "val_dataset.json"),  # или train_dataset.json
        
        # Если нужно создать из CSV (если есть)
        # "csv_path": str(DATASET_ROOT / "some_file.csv"),
        # "image_root": str(DATASET_ROOT / "reference_images"),
        
        "max_samples": 100,  # Для теста, уберите для полной валидации
        "season_filter": None,
        "output_results": str(Path(__file__).parent / "validation_results.json")
    }
    
    # Проверяем существование файлов
    print(f"\n📁 Поиск датасета...")
    print(f"   Корень датасета: {DATASET_ROOT}")
    print(f"   Существует: {DATASET_ROOT.exists()}")
    
    if DATASET_ROOT.exists():
        print(f"\n📂 Содержимое {DATASET_ROOT}:")
        for item in DATASET_ROOT.iterdir():
            print(f"   - {item.name}")
    
    # Проверяем JSON файлы
    json_files = list(DATASET_ROOT.glob("*.json"))
    if json_files:
        print(f"\n📄 Найдены JSON файлы:")
        for jf in json_files:
            print(f"   - {jf.name} ({jf.stat().st_size / 1024:.1f} KB)")
        
        # Используем первый найденный JSON если не указан конкретный
        if not os.path.exists(VALIDATION_CONFIG["data_path"]) and json_files:
            VALIDATION_CONFIG["data_path"] = str(json_files[0])
            print(f"\n✅ Используем: {VALIDATION_CONFIG['data_path']}")
    
    # Проверяем папку с изображениями
    images_dir = DATASET_ROOT / "reference_images"
    if images_dir.exists():
        print(f"\n📸 Папка с изображениями: {images_dir}")
        for season_dir in images_dir.iterdir():
            if season_dir.is_dir():
                img_count = len(list(season_dir.glob("*.*")))
                print(f"   - {season_dir.name}: {img_count} изображений")
    
    validator = ModelValidator(cfg)
    
    if not os.path.exists(VALIDATION_CONFIG["data_path"]):
        print(f"\n❌ Файл датасета не найден: {VALIDATION_CONFIG['data_path']}")
        print("\n📝 Создаем датасет из структуры папок...")
        
        # Создаем датасет из папок
        create_dataset_from_folders(
            images_dir=str(images_dir),
            output_json=VALIDATION_CONFIG["data_path"]
        )
    
    # Запуск валидации
    async def run_validation():
        metrics = await validator.validate_dataset(
            data_path=VALIDATION_CONFIG["data_path"],
            max_samples=VALIDATION_CONFIG["max_samples"],
            season_filter=VALIDATION_CONFIG["season_filter"]
        )
        
        validator.print_report()
        validator._save_results(validator.results, VALIDATION_CONFIG["output_results"])
        print(f"\n💾 Результаты сохранены в {VALIDATION_CONFIG['output_results']}")
        
        return metrics
    
    metrics = asyncio.run(run_validation())
    
    # Сохраняем метрики
    metrics_path = Path(VALIDATION_CONFIG["output_results"]).parent / "validation_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Метрики сохранены в {metrics_path}")


def create_dataset_from_folders(images_dir: str, output_json: str):
    """
    Создает датасет из структуры папок:
    reference_images/
        winter/
            image1.jpg
            image2.jpg
        spring/
        summer/
        autumn/
    """
    images_path = Path(images_dir)
    data = []
    
    season_map = {
        'winter': 'winter',
        'spring': 'spring', 
        'summer': 'summer',
        'autumn': 'autumn',
        'fall': 'autumn'
    }
    
    for season_folder in images_path.iterdir():
        if not season_folder.is_dir():
            continue
        
        season_name = season_folder.name.lower()
        season = season_map.get(season_name, season_name)
        
        if season not in ['winter', 'spring', 'summer', 'autumn']:
            print(f"⚠️ Пропускаем неизвестную папку: {season_name}")
            continue
        
        images = list(season_folder.glob("*.jpg")) + list(season_folder.glob("*.png")) + list(season_folder.glob("*.jpeg"))
        
        for img_path in images:
            data.append({
                "image": str(img_path),
                "season": season,
                "lat": None,
                "lon": None,
                "city": None
            })
        
        print(f"   {season}: {len(images)} изображений")
    
    # Сохраняем JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Создан датасет: {len(data)} образцов")
    print(f"   Сохранен в: {output_json}")
    
    return output_json

if __name__ == "__main__":
    # Меняем рабочую директорию на корень проекта
    os.chdir(str(ROOT_DIR))
    main()