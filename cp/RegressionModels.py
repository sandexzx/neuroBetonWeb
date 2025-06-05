import torch
from torchvision.models import mobilenet_v2
import torch.nn as nn
import cv2
import numpy as np
import math
import torch.nn.functional as F
from skimage.feature import hog
from torchvision import models

class SmallCNNRegressor(nn.Module):
    def __init__(self):
        super(SmallCNNRegressor, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),  # Assumes input size of 224x224
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class CannyEdgeFeatureExtractor(nn.Module):
    def __init__(self, out_features=16):
        super().__init__()
        self.out_features = out_features
        self.projector = None  # lazy init

    def extract_features(self, img_tensor):
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edge_ratio = np.sum(edges > 0) / edges.size
        mean_edge_val = np.mean(edges)

        # Divide into 4x4 grid and compute edge density
        h, w = edges.shape
        cell_h, cell_w = h // 4, w // 4
        grid_features = []
        for i in range(4):
            for j in range(4):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                grid_features.append(np.sum(cell > 0) / cell.size)
        
        feature_vec = np.array([edge_ratio, mean_edge_val / 255.0] + grid_features)
        feature_vec = (feature_vec - np.mean(feature_vec)) / (np.std(feature_vec) + 1e-8)
        return feature_vec.astype(np.float32)

    def forward(self, x):
        features = np.array([self.extract_features(img) for img in x])
        feats = torch.tensor(features).to(x.device)

        if self.projector is None:
            in_dim = feats.shape[1]
            self.projector = nn.Sequential(
                nn.Linear(in_dim, self.out_features),
                nn.ReLU()
            ).to(x.device)

        return self.projector(feats)


class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        model = mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.projector(x)
    

class CombinedCNN_CannyRegressor(nn.Module):
    def __init__(self, cnn_extractor, edge_extractor, regressor_dim=64):
        super().__init__()
        self.cnn = cnn_extractor
        self.edge = edge_extractor

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            cnn_dim = self.cnn(dummy).shape[1]
            edge_dim = self.edge(dummy).shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(cnn_dim + edge_dim, regressor_dim),
            nn.ReLU(),
            nn.Linear(regressor_dim, 1)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        edge_feat = self.edge(x)
        combined = torch.cat([cnn_feat, edge_feat], dim=1)
        return self.regressor(combined)
    
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class TextureFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.extractor(x)
        return x.view(x.size(0), -1)

class CombinedRegressor(nn.Module):
    def __init__(self, cnn_extractor, texture_extractor, hidden_dim=128):
        super().__init__()
        self.cnn = cnn_extractor
        self.texture = texture_extractor
        
        # Calculate feature size
        dummy_input = torch.randn(1, 3, 224, 224)
        feat_dim = self.cnn(dummy_input).shape[1] + self.texture(dummy_input).shape[1]
        
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        cnn_feat = self.cnn(x)
        texture_feat = self.texture(x)
        combined = torch.cat([cnn_feat, texture_feat], dim=1)
        return self.regressor(combined).squeeze(1)
    
class GaborTextureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, kernel_size=11):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self._init_gabor_weights()

    def _init_gabor_weights(self):
        # Initialize filters with Gabor-like weights
        for i in range(self.conv.out_channels):
            theta = i * math.pi / self.conv.out_channels
            sigma = 2.0
            lamda = 10.0
            kernel = self._gabor_kernel(self.conv.kernel_size[0], sigma, theta, lamda)
            # self.conv.weight.data[i, 0] = torch.tensor(kernel, dtype=torch.float32)
            self.conv.weight.data[i, 0] = kernel.clone().detach()
        if self.conv.in_channels > 1:
            for i in range(1, self.conv.in_channels):
                self.conv.weight.data[:, i] = self.conv.weight.data[:, 0]

    def _gabor_kernel(self, ksize, sigma, theta, lamda):
        half = ksize // 2
        y, x = torch.meshgrid(torch.arange(-half, half+1), torch.arange(-half, half+1), indexing='ij')
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        gb = torch.exp(-(x_theta**2 + y_theta**2) / (2*sigma**2)) * torch.cos(2*math.pi * x_theta / lamda)
        return gb

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)
    
class LinearRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)
    
class MLPRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)
    
class CVFeatureExtractor(nn.Module):
    def __init__(self, out_features=32):
        super().__init__()
        self.out_features = out_features
        self.projector = None  # Delay initialization

    def extract_cv_features(self, img_tensor):

        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # HOG features (partial to reduce size)
        hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        hog_features = hog_features[:64]  # Limit to reduce dimensionality

        # RGB Histograms
        hist_r = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [16], [0, 256]).flatten()
        hist_b = cv2.calcHist([img], [2], None, [16], [0, 256]).flatten()

        # Edge strength
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edge_mean = np.mean(np.sqrt(sobelx**2 + sobely**2))

        # Combine features
        feature_vec = np.concatenate([hog_features, hist_r, hist_g, hist_b, [edge_mean]])

        # Normalize
        feature_vec = (feature_vec - np.mean(feature_vec)) / (np.std(feature_vec) + 1e-8)
        return feature_vec.astype(np.float32)

    def forward(self, x):
        batch_feats = []
        for img in x:
            feat = self.extract_cv_features(img)
            batch_feats.append(feat)
        feats = torch.tensor(np.array(batch_feats)).to(x.device)

        # Lazy init of projector
        if self.projector is None:
            in_dim = feats.shape[1]
            self.projector = nn.Sequential(
                nn.Linear(in_dim, self.out_features),
                nn.ReLU()
            ).to(x.device)

        return self.projector(feats)
    
class CombinedRegressorModular(nn.Module):
    def __init__(self, cnn_extractor, texture_extractor, regressor_class, hidden_dim=128):
        super().__init__()
        self.cnn = cnn_extractor
        self.texture = texture_extractor

        dummy_input = torch.randn(1, 3, 224, 224)
        feat_dim = self.cnn(dummy_input).shape[1] + self.texture(dummy_input).shape[1]

        if regressor_class == "mlp":
            self.regressor = MLPRegressor(feat_dim, hidden_dim)
        elif regressor_class == "linear":
            self.regressor = LinearRegressor(feat_dim)
        else:
            raise ValueError("Unsupported regressor_class")

    def forward(self, x):
        cnn_feat = self.cnn(x)
        texture_feat = self.texture(x)
        combined = torch.cat([cnn_feat, texture_feat], dim=1)
        return self.regressor(combined)
    
class CombinedRegressorFull(nn.Module):
    def __init__(self, cnn_extractor, texture_extractor, cv_extractor,
                 regressor_class="mlp", hidden_dim=128):
        super().__init__()
        self.cnn = cnn_extractor
        self.texture = texture_extractor
        self.cv = cv_extractor

        # Dummy input to get dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        feat_cnn = self.cnn(dummy_input).shape[1]
        feat_tex = self.texture(dummy_input).shape[1]
        feat_cv = self.cv(dummy_input).shape[1]

        combined_dim = feat_cnn + feat_tex + feat_cv

        if regressor_class == "mlp":
            self.regressor = MLPRegressor(combined_dim, hidden_dim)
        elif regressor_class == "linear":
            self.regressor = LinearRegressor(combined_dim)
        else:
            raise ValueError("Unsupported regressor type")

    def forward(self, x):
        cnn_feat = self.cnn(x)
        tex_feat = self.texture(x)
        cv_feat = self.cv(x)
        combined = torch.cat([cnn_feat, tex_feat, cv_feat], dim=1)
        return self.regressor(combined)
    

class HOGFeatureExtractor(nn.Module):
    def __init__(self, out_features=32):
        super().__init__()
        self.out_features = out_features
        self.projector = None  # Lazy init

    def extract_features(self, img_tensor):
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # Extract HOG features
        hog_features = hog(
            gray,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        hog_features = hog_features[:128]  # limit dimensionality

        # Normalize
        hog_features = (hog_features - np.mean(hog_features)) / (np.std(hog_features) + 1e-8)
        return hog_features.astype(np.float32)

    def forward(self, x):
        features = np.array([self.extract_features(img) for img in x])
        feats = torch.tensor(features).to(x.device)

        if self.projector is None:
            in_dim = feats.shape[1]
            self.projector = nn.Sequential(
                nn.Linear(in_dim, self.out_features),
                nn.ReLU()
            ).to(x.device)

        return self.projector(feats)


class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    

class CracksRecognitionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'CracksRecognitionModel'
        # first layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu = nn.ReLU(inplace=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32, )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # second layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.dropout1 = nn.Dropout(0.3)
        self.pool2 = self.pool1
        # third layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
        self.dropout2 = self.dropout1
        self.pool3 = self.pool1
        # forth layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same')
        self.dropout4 = self.dropout1
        self.pool4 = self.pool1

        self.linear1 = nn.Linear(in_features=50176, out_features=1024)
        self.dropout = nn.Dropout(0.5)

        self.linear2 = nn.Linear(in_features=1024, out_features=2)
        # self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        # 1st layer
        x = self.pool1(self.batchnorm1(self.relu(self.conv1(x))))
        # 2nd layer
        x = self.pool2(self.dropout1(self.relu(self.conv2(x))))
        # 3rd layer
        x = self.pool3(self.dropout2(self.relu(self.conv3(x))))
        # 4th layer
        x = self.pool4(self.dropout4(self.relu(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        # x = self.sigmoid(x)
        # x = torch.squeeze(x)
        return x


# Predefined Models
CR_CNNFE_TFE_model = CombinedRegressor(CNNFeatureExtractor(), TextureFeatureExtractor(), hidden_dim=128)
CR_CNNFE_TFE_model.name = 'CR_CNNFE_TFE'

CRM_CNNFE_TFE_mlp_model = CombinedRegressorModular(CNNFeatureExtractor(), TextureFeatureExtractor(), regressor_class='mlp', hidden_dim=128)
CRM_CNNFE_TFE_mlp_model.name = 'CRM_CNNFE_TFE_mlp'

CRM_CNNFE_GTE_linear_model = CombinedRegressorModular(CNNFeatureExtractor(), GaborTextureExtractor(), regressor_class='linear', hidden_dim=128)
CRM_CNNFE_GTE_linear_model.name = 'CRM_CNNFE_GTE_linear'

CRM_MNFE_TFE_mlp_model = CombinedRegressorModular(MobileNetFeatureExtractor(), TextureFeatureExtractor(), regressor_class='mlp', hidden_dim=128)
CRM_MNFE_TFE_mlp_model.name = 'CRM_MNFE_TFE_mlp'

CRF_CNNFE_TFE_CVFE_mlp_model = CombinedRegressorFull(CNNFeatureExtractor(), TextureFeatureExtractor(), CVFeatureExtractor(), regressor_class='mlp', hidden_dim=128)
CRF_CNNFE_TFE_CVFE_mlp_model.name = 'CRF_CNNFE_TFE_CVFE_mlp'

CRF_MNFE_GTE_CVFE_linear_model = CombinedRegressorFull(MobileNetFeatureExtractor(), GaborTextureExtractor(), CVFeatureExtractor(), regressor_class='linear', hidden_dim=128)
CRF_MNFE_GTE_CVFE_linear_model.name = 'CRF_MNFE_GTE_CVFE_linear'

CCannyR_MNFE_CEFE_model = CombinedCNN_CannyRegressor(MobileNetFeatureExtractor(), CannyEdgeFeatureExtractor(out_features=16), regressor_dim=64)
CCannyR_MNFE_CEFE_model.name = 'CCannyR_MNFE_CEFE'

CCannyR_MNFE_HOG_model = CombinedCNN_CannyRegressor(MobileNetFeatureExtractor(), HOGFeatureExtractor(out_features=32), regressor_dim=64)
CCannyR_MNFE_HOG_model.name = 'CCannyR_MNFE_HOG'

CCannyR_CNNFE_HOG_model = CombinedCNN_CannyRegressor(CNNFeatureExtractor(), HOGFeatureExtractor(out_features=32), regressor_dim=64)
CCannyR_CNNFE_HOG_model.name = 'CCannyR_CNNFE_HOG'