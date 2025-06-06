import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import yaml
import pickle
import logging
from models.cracks_model import CracksRecognitionModel
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
            self.cracks_model.load_state_dict(torch.load('models/cracks/best_cracks_detection_model.pt'))
            self.cracks_model.to(self.device)
            self.cracks_model.eval()
            self.logger.info("Модель определения трещин успешно инициализирована")
            
            # Инициализация модели классификации типа бетона
            self.classification_model = ClassificationModel(num_classes=5)
            self.classification_model.load_state_dict(torch.load('models/classification/best_concrete_type_classification_model.pt'))
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
            
            # Делаем предсказание
            with torch.no_grad():
                prediction = self.cracks_model(image_tensor)
                logger.info(f"Raw prediction: {prediction.cpu().numpy()}")
                # Применяем softmax для получения вероятностей
                probabilities = F.softmax(prediction, dim=1)
                # Получаем предсказанный класс и вероятность
                predicted_class = torch.argmax(probabilities, dim=1).item()
                crack_probability = probabilities[0][0].item()  # вероятность класса "есть трещина"
                logger.info(f"Predicted class: {predicted_class}, Probability: {crack_probability}")
                
            # 0 - есть трещина, 1 - нет трещины
            result = {
                'has_cracks': predicted_class == 0,
                'crack_probability': crack_probability
            }
            logger.info(f"Final result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in crack prediction: {str(e)}")
            raise
    
    def predict_concrete_type(self, image_path):
        if not self.classification_model:
            self._initialize_models()
            
        try:
            logger.info("Predicting concrete type...")
            # Загружаем и преобразуем изображение
            logger.info("Loading and transforming image...")
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.to(self.device)
            logger.info(f"Image tensor shape: {image_tensor.shape}")
            
            # Делаем предсказание
            logger.info("Making prediction...")
            with torch.no_grad():
                prediction = self.classification_model(image_tensor)
                logger.info(f"Raw prediction shape: {prediction.shape}")
                logger.info(f"Raw prediction values: {prediction}")
                
                # Применяем softmax к логам
                probabilities = F.softmax(prediction, dim=1)
                logger.info(f"Probabilities shape: {probabilities.shape}")
                logger.info(f"Probabilities values: {probabilities}")
                
                predicted_class = torch.argmax(probabilities, dim=1).item()
                logger.info(f"Predicted class index: {predicted_class}")
                
                confidence = probabilities[0][predicted_class].item()
                logger.info(f"Confidence: {confidence}")
                
            # Используем маппинг для получения названия типа бетона
            concrete_type = self.label_mapping[predicted_class]
            logger.info(f"Concrete type prediction: {concrete_type} (confidence: {confidence})")
                
            return {
                'concrete_type': concrete_type,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in concrete type prediction: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Создаем экземпляр ModelService для использования в других модулях
model_service = ModelService() 