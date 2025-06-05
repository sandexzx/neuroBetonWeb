import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class MobileNetRegressor(nn.Module):
    """
    MobileNet-based регрессор для предсказания прочности бетона
    Поддерживает гибридные features: CNN + Canny edges
    """
    
    def __init__(self,
                 backbone: str = "mobilenet_v2",
                 pretrained: bool = True,
                 use_canny: bool = True,
                 dropout: float = 0.3,
                 num_classes: int = 1):
        """
        Args:
            backbone: тип MobileNet (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large)
            pretrained: использовать предобученные веса
            use_canny: использовать Canny edges как дополнительные features
            dropout: dropout rate
            num_classes: количество выходов (1 для регрессии)
        """
        super(MobileNetRegressor, self).__init__()
        
        self.use_canny = use_canny
        self.backbone_name = backbone
        
        # CNN backbone для RGB изображений
        self.cnn_backbone = self._create_backbone(backbone, pretrained)
        cnn_features = self._get_backbone_features(backbone)
        
        # Canny branch (если используется)
        if use_canny:
            self.canny_branch = self._create_canny_branch()
            canny_features = 128  # из canny_branch
            total_features = cnn_features + canny_features
        else:
            total_features = cnn_features
        
        # Регрессионная голова
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Создание CNN backbone"""
        if backbone == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
            # Убираем classifier, оставляем только features
            return model.features
        
        elif backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=pretrained)
            return model.features
        
        elif backbone == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(pretrained=pretrained)
            return model.features
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _get_backbone_features(self, backbone: str) -> int:
        """Получение количества features из backbone"""
        if backbone == "mobilenet_v2":
            return 1280
        elif backbone == "mobilenet_v3_small":
            return 576
        elif backbone == "mobilenet_v3_large":
            return 960
        else:
            raise ValueError(f"Unknown feature count for {backbone}")
    
    def _create_canny_branch(self) -> nn.Module:
        """Создание ветки для обработки Canny edges"""
        return nn.Sequential(
            # Первый блок
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Второй блок
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Третий блок
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # Фиксированный размер для concat
        )
    
    def forward(self, rgb_image: torch.Tensor, canny_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            rgb_image: RGB изображение [B, 3, H, W]
            canny_edges: Canny edges [B, 1, H, W] (опционально)
            
        Returns:
            predictions: предсказания прочности [B, 1]
        """
        # CNN features из RGB
        cnn_features = self.cnn_backbone(rgb_image)  # [B, channels, H', W']
        
        if self.use_canny and canny_edges is not None:
            # Canny features
            canny_features = self.canny_branch(canny_edges)  # [B, 128, 8, 8]
            
            # Приводим cnn_features к размеру 8x8 для concat
            cnn_features_resized = F.adaptive_avg_pool2d(cnn_features, (8, 8))
            
            # Объединяем features по каналам
            combined_features = torch.cat([cnn_features_resized, canny_features], dim=1)
        else:
            combined_features = cnn_features
        
        # Регрессия
        output = self.regressor(combined_features)
        
        return output 