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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.initialized = False
        self.model = None
        self.crack_model = None
        self.classification_model = None
        self.transform = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapping = None
        
    def initialize(self):
        if self.initialized:
            return
            
        try:
            # Load config
            logger.info("Loading config...")
            with open('model/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Initialize transforms
            logger.info("Initializing transforms...")
            self.transform = transforms.Compose([
                transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Load label mapping for concrete type classification
            logger.info("Loading label mapping...")
            with open('models/classification/label_mapping.pkl', 'rb') as f:
                loaded_mapping = pickle.load(f)
            # Создаем обратный маппинг (индекс -> название класса)
            self.label_mapping = {idx: label for label, idx in loaded_mapping.items()}
            logger.info(f"Loaded label mapping: {self.label_mapping}")
            
            # Initialize strength prediction model
            logger.info("Initializing strength prediction model...")
            self.model = MobileNetRegressor(
                backbone=self.config['model']['backbone'],
                pretrained=False,
                use_canny=self.config['model']['use_hybrid_features'],  # Используем значение из конфига
                dropout=self.config['model']['dropout']
            )
            checkpoint = torch.load('model/best_model.pth', map_location=self.device)  # Используем ту же модель
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logger.info("Strength prediction model initialized successfully")
            
            # Initialize crack detection model
            logger.info("Initializing crack detection model...")
            self.crack_model = CracksRecognitionModel()
            self.crack_model.load_state_dict(torch.load('models/cracks/best_CracksRecognitionModel_model.pt', map_location=self.device))
            self.crack_model.to(self.device)
            self.crack_model.eval()
            logger.info("Crack detection model initialized successfully")
            
            # Initialize concrete type classification model
            logger.info("Initializing classification model...")
            self.classification_model = ClassificationModel(num_classes=len(self.label_mapping))
            checkpoint = torch.load('models/classification/best_ClassificationModel_model.pt', map_location=self.device)
            logger.info(f"Classification model checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                logger.info("Loading model_state_dict from checkpoint")
                self.classification_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.info("Loading checkpoint directly as state_dict")
                self.classification_model.load_state_dict(checkpoint)
            self.classification_model.to(self.device)
            self.classification_model.eval()
            logger.info("Classification model initialized successfully")
            
            self.initialized = True
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
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
        if not self.initialized:
            self.initialize()
            
        try:
            logger.info("Predicting strength...")
            # Загружаем и преобразуем изображение
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.to(self.device)
            
            # Получаем Canny edges
            canny_edges = self.get_canny_edges(image_path, self.config['data']['image_size'])
            canny_edges = canny_edges.to(self.device)
            
            # Делаем предсказание
            with torch.no_grad():
                prediction = self.model(image_tensor, canny_edges)
                logger.info(f"Raw prediction: {prediction.cpu().numpy()}")
                
            # Модель уже выдает значения в МПа
            strength = prediction.cpu().numpy()[0][0]
            logger.info(f"Final strength prediction: {strength}")
            
            return strength
            
        except Exception as e:
            logger.error(f"Error in strength prediction: {str(e)}")
            raise
    
    def predict_cracks(self, image_path):
        if not self.initialized:
            self.initialize()
            
        try:
            logger.info("Predicting cracks...")
            # Загружаем и преобразуем изображение
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.to(self.device)
            
            # Делаем предсказание
            with torch.no_grad():
                prediction = self.crack_model(image_tensor)
                probabilities = torch.exp(prediction)  # Convert log probabilities to probabilities
                crack_probability = probabilities[0][1].item()  # Probability of crack
                
            logger.info(f"Crack prediction: {crack_probability}")
            
            return {
                'has_cracks': crack_probability > 0.5,
                'crack_probability': crack_probability
            }
            
        except Exception as e:
            logger.error(f"Error in crack prediction: {str(e)}")
            raise
    
    def predict_concrete_type(self, image_path):
        if not self.initialized:
            self.initialize()
            
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
                
                probabilities = torch.exp(prediction)  # Convert log probabilities to probabilities
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