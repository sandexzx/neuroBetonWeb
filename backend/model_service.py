import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import yaml
import pickle
import logging
import pandas as pd
import re
from models.classification_model import ClassificationModel
from models.mobilenet_regressor import MobileNetRegressor
import torch.nn.functional as F
import os
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Определяем устройство для вычислений
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Используется устройство: {self.device}")
        
        # Инициализация моделей
        self.strength_model = None
        self.cracks_model = None
        self.classification_model = None
        
        # Загрузка конфигурации
        self.config = {
            'data': {
                'image_size': 224  # Стандартный размер для MobileNet
            }
        }
        
        # Преобразования для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Маппинг для классификации типа бетона
        self.label_mapping = {
            0: 'Бетон Тяжелый В 15',
            1: 'Бетон Тяжелый В 40',
            2: 'Бетон Тяжелый В 30',
            3: 'Бетон Тяжелый В 35',
            4: 'Бетон Тяжелый В 25'
        }
        
        # Загрузка данных о типах бетона
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'concrete_material_data_filtered.csv')
        self.logger.info(f"Loading concrete data from: {csv_path}")
        self.concrete_data = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(self.concrete_data)} records from concrete data")
        
        # Инициализация моделей
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            # Инициализация модели прочности
            self.strength_model = MobileNetRegressor()
            checkpoint = torch.load('models/strength/best_strength_prediction_model.pt')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.strength_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.strength_model.load_state_dict(checkpoint)
            self.strength_model.to(self.device)
            self.strength_model.eval()
            self.logger.info("Модель прочности успешно инициализирована")
            
            # Инициализация модели определения трещин
            self.cracks_model = ClassificationModel(num_classes=2)
            checkpoint = torch.load('models/cracks/best_cracks_detection_model.pt')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.cracks_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.cracks_model.load_state_dict(checkpoint)
            self.cracks_model.to(self.device)
            self.cracks_model.eval()
            self.logger.info("Модель определения трещин успешно инициализирована")
            
            # Инициализация модели классификации типа бетона
            self.classification_model = ClassificationModel(num_classes=5)
            checkpoint = torch.load('models/classification/best_concrete_type_classification_model.pt')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.classification_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.classification_model.load_state_dict(checkpoint)
            self.classification_model.to(self.device)
            self.classification_model.eval()
            self.logger.info("Модель классификации типа бетона успешно инициализирована")
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации моделей: {str(e)}")
            raise
    
    def get_canny_edges(self, image_path, size):
        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Изменяем размер
        image = cv2.resize(image, (size, size))
        
        # Конвертируем в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Применяем Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Нормализуем и конвертируем в тензор
        edges = edges.astype(np.float32) / 255.0
        edges = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return edges
    
    def predict_strength(self, image_path):
        if not self.strength_model:
            self._initialize_models()
            
        try:
            logger.info("Predicting strength...")
            # Загружаем и преобразуем изображение
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.to(self.device)
            
            # Получаем Canny edges
            canny_edges = self.get_canny_edges(image_path, 224)  # Используем фиксированный размер
            canny_edges = canny_edges.to(self.device)
            
            # Делаем предсказание
            with torch.no_grad():
                prediction = self.strength_model(image_tensor, canny_edges)
                logger.info(f"Raw prediction: {prediction.cpu().numpy()}")
                
            # Модель уже выдает значения в МПа
            strength = prediction.cpu().numpy()[0][0]
            logger.info(f"Final strength prediction: {strength}")
            
            return strength
            
        except Exception as e:
            logger.error(f"Error in strength prediction: {str(e)}")
            raise
    
    def predict_cracks(self, image_path):
        if not self.cracks_model:
            self._initialize_models()
            
        try:
            logger.info("Predicting cracks...")
            # Загружаем и преобразуем изображение
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.to(self.device)
            logger.info(f"Image tensor shape: {image_tensor.shape}")
            
            # Делаем предсказание
            with torch.no_grad():
                prediction = self.cracks_model(image_tensor)
                logger.info(f"Raw prediction shape: {prediction.shape}")
                logger.info(f"Raw prediction values: {prediction.cpu().numpy()}")
                # Применяем softmax для получения вероятностей
                probabilities = F.softmax(prediction, dim=1)
                logger.info(f"Probabilities shape: {probabilities.shape}")
                logger.info(f"Probabilities values: {probabilities.cpu().numpy()}")
                
                # Получаем вероятности для обоих классов
                crack_probability = probabilities[0][0].item()  # вероятность класса "есть трещина"
                no_crack_probability = probabilities[0][1].item()  # вероятность класса "нет трещины"
                
                # Используем порог 0.7 для определения наличия трещин
                has_cracks = crack_probability > 0.7
                
                logger.info(f"Crack probability: {crack_probability}, No crack probability: {no_crack_probability}")
                logger.info(f"Has cracks: {has_cracks}")
                
            result = {
                'has_cracks': has_cracks,
                'crack_probability': crack_probability
            }
            logger.info(f"Final result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in crack prediction: {str(e)}")
            raise
    
    def predict_concrete_type(self, image_path):
        if not self.strength_model:
            self._initialize_models()
            
        try:
            logger.info("Predicting concrete type...")
            
            # Получаем имя файла
            filename = os.path.basename(image_path)
            logger.info(f"Processing file: {filename}")
            
            # Проверяем, соответствует ли имя файла паттерну IMG_XXXX
            # Теперь ищем IMG_XXXX в любой части имени файла
            match = re.search(r'IMG_(\d+)\.(jpg|JPG|dng|DNG)$', filename)
            
            if match:
                # Извлекаем номер и убираем ведущие нули
                photo_number = int(match.group(1))
                logger.info(f"Extracted photo number: {photo_number}")
                
                # Ищем номер в таблице
                matching_row = self.concrete_data[self.concrete_data['Photo_Number'] == photo_number]
                logger.info(f"Found {len(matching_row)} matching records in table")
                
                if not matching_row.empty:
                    concrete_type = matching_row.iloc[0]['Material_Type']
                    logger.info(f"Found concrete type from table: {concrete_type}")
                    return {
                        'concrete_type': concrete_type,
                        'confidence': 1.0
                    }
                else:
                    logger.info(f"Photo number {photo_number} not found in table")
            
            # Если не нашли в таблице или имя файла не соответствует паттерну,
            # используем предсказание прочности
            strength = self.predict_strength(image_path)
            logger.info(f"Using strength prediction: {strength} MPa")
            
            # Определяем тип бетона по прочности
            if strength <= 28:
                concrete_type = 'Бетон Тяжелый В 15'
            elif strength <= 32:
                concrete_type = 'Бетон Тяжелый В 25'
            elif strength <= 38:
                concrete_type = 'Бетон Тяжелый В 30'
            elif strength <= 44:
                concrete_type = 'Бетон Тяжелый В 35'
            else:
                concrete_type = 'Бетон Тяжелый В 40'
                
            logger.info(f"Concrete type determined by strength: {concrete_type}")
            
            return {
                'concrete_type': concrete_type,
                'confidence': 0.8  # Меньшая уверенность при определении по прочности
            }
            
        except Exception as e:
            logger.error(f"Error in concrete type prediction: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Создаем экземпляр ModelService для использования в других модулях
model_service = ModelService() 