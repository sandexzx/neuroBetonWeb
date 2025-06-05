import os
import os.path
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns   
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch.nn.init as init
import cv2
from skimage.feature import hog
from torch.utils.data import Subset, ConcatDataset
from sklearn.metrics import r2_score
from torchvision.io import read_image
from typing import Optional, Callable, Union, List, Dict, Any, Tuple, Literal



class ImageDatasetIPKON(Dataset):
    """
    A PyTorch Dataset for loading images and labels from an Excel file for either
    concrete type classification or pressure regression tasks.

    Supports normalization for regression targets and optional image transformations.
    """
    
    def __init__(
        self,
        excel_path: str,
        image_folder: str,
        dataset_type: str,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        one_concrete_type: bool = True
    ):
        """
        Initialize the ImageDatasetIPKON dataset.

        Args:
            excel_path (str): Path to the Excel file containing metadata and labels.
            image_folder (str): Directory where the images are stored.
            dataset_type (str): Either 'classification' or 'regression'.
            transform (Callable, optional): Transformations to apply to the images.
            normalize (bool): Whether to normalize regression values (default: True).
            one_concrete_type (bool): Whether to filter to one concrete type in regression (default: True).

        Raises:
            ValueError: If an invalid dataset_type is provided.
        """
        allowed_dataset_types = {"classification", "regression"}
        if dataset_type not in allowed_dataset_types:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Choose from {allowed_dataset_types}.")

        self.dataset_type = dataset_type
        self.excel_path = excel_path
        self.image_folder = image_folder
        self.transform = transform
        self.normalize = normalize

        self.labels_df = pd.read_excel(self.excel_path)
        self.labels_df.loc[:, 'photo_name'] = self.labels_df.photo_idx.apply(lambda x: f"IMG_0{x}.jpg")


        if self.dataset_type == 'classification':
            self.label_mapping: Dict[str, int] = {
                label: idx for idx, label in enumerate(self.labels_df['material_type'].unique())
            }
            self.labels_df['material_encoded'] = self.labels_df['material_type'].map(self.label_mapping)
            self.labels_col = 'material_encoded'

        if self.dataset_type == 'regression':
            if one_concrete_type:
                self.labels_df = self.labels_df[self.labels_df['material_type'] == "Бетон Тяжелый В 15"].copy()
            if self.normalize:
                self.scaler = MinMaxScaler()
                self.labels_df.loc[:, 'normalized_strength'] = self.scaler.fit_transform(self.labels_df['strength'].values.reshape(-1, 1))
            else:
                self.scaler = None
                self.labels_df.loc[:, 'normalized_strength'] = self.labels_df.strength
            self.labels_col = 'normalized_strength'

        self.image_paths: List[str] = []
        self.labels: List[Union[int, float]] = []
        
        for _, row in self.labels_df.iterrows():
            photo_name = row['photo_name']
            photo_path = os.path.join(self.image_folder, photo_name)

            if os.path.isfile(photo_path):
                self.image_paths.append(photo_path)
                self.labels.append(row[self.labels_col])
            else:
                print(f"Warning: File {photo_name} not found in {image_folder}")

        
    def __len__(self) -> int:
        """
        Return the number of valid image-label pairs in the dataset.
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve a single image-label pair.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[Tensor, Union[int, Tensor]]: Image tensor and label.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.dataset_type == 'regression':
            return image, torch.tensor(label, dtype=torch.float32)

        if self.dataset_type == 'classification':
            return image, label
    
    def reverse_normalization(self, strength_value: float) -> float:
        """
        Convert a normalized strength value back to its original scale.

        Args:
            strength_value (float): Normalized strength value.

        Returns:
            float: Original (unnormalized) strength value.
        """
        if self.normalize and self.scaler:
            return self.scaler.inverse_transform(np.array(strength_value).reshape(1, -1)).item()
        else:
            return strength_value
    
    def get_values_scaler(self) -> Optional[MinMaxScaler]:
        """
        Get the MinMaxScaler used for normalization (if any).

        Returns:
            MinMaxScaler or None
        """
        return self.scaler if self.dataset_type == 'regression' else None
    
    def get_label_mapping(self) -> Optional[Dict[str, int]]:
        """
        Get the label-to-index mapping for classification.

        Returns:
            dict or None: Mapping of material type to encoded label.
        """
        return self.label_mapping if self.dataset_type == 'classification' else None
    

class ImageDatasetSDNET(Dataset):
    """
    A PyTorch Dataset for the SDNET dataset used for binary classification
    of cracked vs. non-cracked concrete images.

    The dataset assumes the directory structure: root/class_name/image.jpg,
    where `class_name` is either 'Cracked' or 'Non-cracked'.
    """
    def __init__(
        self,
        img_dir: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize the ImageDatasetSDNET dataset.

        Args:
            img_dir (Path): Path to the root directory containing images organized by class subfolders.
            transform (Callable, optional): Transformations to apply to the images.
            target_transform (Callable, optional): Transformations to apply to the labels.
        """
        self.imgs_path: List[Path] = list(img_dir.glob(r'*/*/*.jpg'))
        
        # label mapping
        self.labels_map: Dict[str, int] = {
            'Cracked': 1,
            'Non-cracked': 0
        }

        # Extract labels from folder structure
        imgs_labels = [self.labels_map[os.path.split(os.path.split(p)[0])[-1]] for p in self.imgs_path]

        self.df_data = pd.DataFrame(
            list(zip(self.imgs_path, imgs_labels)),
            columns=['JPG', 'CATEGORY']
        )

        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = 'classification'

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.imgs_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
        """
        Retrieve a single image-label pair.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, int or Tensor]: The image and its label.
        """
        img_path = self.df_data.iloc[idx, 0]
        label = self.df_data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_label_mapping(self) -> Dict[str, int]:
        """
        Get the class-to-index label mapping.

        Returns:
            dict: Mapping from class name to label integer.
        """
        return self.labels_map


def create_dataloaders_fixed(
    excel_path: str,
    image_folder: str,
    dataset_type: str,
    batch_size: int = 4,
    train_split: float = 0.8,
    num_workers: int = 4,
    seed: int = 42,
    one_concrete_type: bool = True,
    train_dataset_multiplier: float = 1,
) -> Tuple[DataLoader, DataLoader, Optional[Dict[str, int]], Optional[object]]:
    """
    Create training and validation DataLoaders for the IPKON image dataset.

    Applies heavy augmentation to the training set and basic resizing to the validation set.
    Can optionally replicate the training dataset to balance size or boost small datasets.

    Args:
        excel_path (str): Path to the Excel file with metadata and labels.
        image_folder (str): Directory containing image files.
        dataset_type (str): 'classification' or 'regression'.
        batch_size (int, optional): Number of samples per batch (default: 4).
        train_split (float, optional): Proportion of the dataset used for training (default: 0.8).
        num_workers (int, optional): Number of worker processes for data loading (default: 4).
        seed (int, optional): Random seed for reproducible splitting (default: 42).
        one_concrete_type (bool, optional): Whether to use only one concrete type for regression (default: True).
        train_dataset_multiplier (float, optional): Multiplier for training data replication (default: 1).

    Returns:
        Tuple[
            DataLoader,         # Training data loader
            DataLoader,         # Validation data loader
            Optional[dict],     # Label mapping (for classification)
            Optional[object]    # Scaler used for normalization (for regression)
        ]
    """

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load full dataset for consistent splitting
    full_dataset = ImageDatasetIPKON(excel_path, image_folder, dataset_type, transform=None, one_concrete_type=one_concrete_type)
    total_size = len(full_dataset)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_size = int(train_split * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Reload datasets with actual transforms
    train_dataset = ImageDatasetIPKON(excel_path, image_folder, dataset_type, transform=train_transform, one_concrete_type=one_concrete_type)
    val_dataset = ImageDatasetIPKON(excel_path, image_folder, dataset_type, transform=val_transform, one_concrete_type=one_concrete_type)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    if train_dataset_multiplier > 1:
        train_subset = ConcatDataset([train_subset] * int(train_dataset_multiplier))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (
        train_loader,
        val_loader,
        full_dataset.get_label_mapping(),
        full_dataset.get_values_scaler()
    )


def create_dataloaders_fixed_cracks(
    img_dir: Path,
    batch_size: int = 4,
    train_split: float = 0.8,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create DataLoaders for the SDNET concrete crack dataset with optional augmentation.

    Splits the dataset into training and validation sets with specified transforms.

    Args:
        img_dir (Path): Path to the root directory of the SDNET dataset.
        batch_size (int, optional): Number of images per batch (default: 4).
        train_split (float, optional): Fraction of data used for training (default: 0.8).
        num_workers (int, optional): Number of subprocesses to use for data loading (default: 4).
        seed (int, optional): Random seed for reproducible train/val splitting (default: 42).

    Returns:
        Tuple[
            DataLoader,       # Training data loader
            DataLoader,       # Validation data loader
            Dict[str, int]    # Label mapping (e.g., {'Cracked': 1, 'Non-cracked': 0})
        ]
    """

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = ImageDatasetSDNET(img_dir, transform=None)
    total_size = len(full_dataset)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_size = int(train_split * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = ImageDatasetSDNET(img_dir, transform=train_transform)
    val_dataset = ImageDatasetSDNET(img_dir, transform=val_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (
        train_loader, 
        val_loader, 
        full_dataset.get_label_mapping()
    )



class BatchViewer:
    """
    A utility class for visualizing batches of images from a PyTorch DataLoader.

    Supports both classification and regression datasets. Automatically resets the
    DataLoader iterator when it reaches the end.

    Args:
        data_loader (DataLoader): The PyTorch DataLoader to visualize from.
        values_scaler (Optional[object], optional): Scaler used to reverse normalization
            for regression targets. Should implement `.inverse_transform()`.
        label_mapping (Optional[Dict[str, int]], optional): Dictionary mapping class labels
            to indices for classification. Will be reversed internally.
    """
    def __init__(
        self,
        data_loader: DataLoader,
        values_scaler: Optional[object] = None,
        label_mapping: Optional[Dict[str, int]] = None
    ) -> None:
        self.data_loader = data_loader
        self.values_scaler = values_scaler
        self.label_mapping = {v: k for k, v in label_mapping.items()} if label_mapping is not None else None  # reverse mapping
        self.iterator = iter(self.data_loader)

        # Detect dataset type (handles Subset)
        if isinstance(self.data_loader.dataset, Subset):
            self.dataset_type = self.data_loader.dataset.dataset.dataset_type
        else:
            self.dataset_type = self.data_loader.dataset.dataset_type

    def __iter__(self):
        return self

    def __next__(self):
        """
        Retrieves the next batch from the DataLoader and visualizes the first 4 images with labels.

        Returns:
            None. Displays a matplotlib plot of the batch.
        """
        try:
            images, labels = next(self.iterator)
        except StopIteration:
            print("Restarting iterator...")
            self.iterator = iter(self.data_loader)
            images, labels = next(self.iterator)

        # Plot batch
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        for i in range(4):
            row = i % 2
            col = i // 2
            img = images[i].permute(1, 2, 0).numpy()

            # Determine label display
            if self.dataset_type == 'classification':
                real_label = self.label_mapping.get(labels[i].item(), labels[i].item()) if self.label_mapping else labels[i].item()

            elif self.dataset_type == 'regression':
                if self.values_scaler:
                    real_label = round(self.values_scaler.inverse_transform(np.array(labels[i].item()).reshape(1, -1)).item(), 1)
                else:
                    real_label = labels[i].item()
            else:
                real_label = labels[i].item()
            
            axs[row, col].imshow(img)
            axs[row, col].set_title(f"Label: {real_label}")
            axs[row, col].axis('off')
        plt.tight_layout()
        plt.show()


# Training & evaluation functions
def train_one_epoch(
    model_type: Literal['classification', 'regression', 'cracks'],
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, Optional[float]]:
    """
    Train the model for one epoch.

    Supports different training modes: classification, regression, and binary crack detection.

    Args:
        model_type (Literal['classification', 'regression', 'cracks']): Type of the model/task.
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to perform training on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, Optional[float]]: Average loss and (for classification/cracks) accuracy, otherwise `None`.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=True):
        if model_type == 'classification' or model_type == 'cracks':
            inputs, labels = inputs.to(device), labels.to(device)
        if model_type == 'regression':
            # inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        if model_type == 'cracks':
            outputs = model(inputs)
        else:
            outputs = model(inputs).squeeze()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        total += inputs.size(0)

        if model_type == 'classification' or model_type == 'cracks':
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            correct_rate = correct / total
        else:
            correct_rate = None

    return running_loss / total, correct_rate

def evaluate(
    model_type: Literal['classification', 'regression', 'cracks'],
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Optional[float], np.ndarray, np.ndarray]:
    """
    Evaluate the model on a validation or test dataset.

    Supports classification, regression, and crack detection tasks.

    Args:
        model_type (Literal['classification', 'regression', 'cracks']): Type of task.
        model (nn.Module): Trained PyTorch model.
        loader (DataLoader): DataLoader for evaluation.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        Tuple[float, Optional[float], np.ndarray, np.ndarray]:
            - Average loss over the dataset
            - Accuracy (only for classification or cracks), otherwise None
            - Predicted labels/values
            - Ground truth labels/values
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            if model_type == 'classification' or model_type == 'cracks':
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
            if model_type == 'regression':
                # inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            if model_type == 'classification' or model_type == 'cracks':
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                # if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            if model_type == 'regression':
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

    if model_type == 'classification' or model_type == 'cracks':
        acc = correct / total
        # if return_preds:
        return running_loss / total, acc, np.array(all_preds), np.array(all_labels)
        # return running_loss / total, acc
    if model_type == 'regression':
        all_preds = torch.cat(all_preds).squeeze().numpy()
        all_labels = torch.cat(all_labels).squeeze().numpy()
        return running_loss / len(loader.dataset), None, all_preds, all_labels
    

def train_model_loop(
    model_type: Literal['classification', 'regression', 'cracks'],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 30
) -> Tuple[List[float], List[float], List[Optional[float]], List[Optional[float]]]:
    """
    Trains and validates a model for a given number of epochs.

    Supports classification, regression, and binary crack detection.

    Args:
        model_type (Literal['classification', 'regression', 'cracks']): Type of task.
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU or GPU).
        num_epochs (int, optional): Number of epochs to train. Defaults to 30.

    Returns:
        Tuple:
            - List of training losses per epoch
            - List of validation losses per epoch
            - List of training accuracies per epoch (or None for regression)
            - List of validation accuracies per epoch (or None for regression)
    """
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_r2 = -float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model_type, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = evaluate(model_type, model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)


        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if model_type == 'classification' or model_type == 'cracks':
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"checkpoints/{model_type}/best_{model.__class__.__name__}_model.pt")
        if model_type == 'regression':
            r2 = r2_score(labels, preds)
            try:
                model_name = model.name
            except:
                model_name = model.__class__.__name__
            if r2 > best_r2:
                best_r2 = r2
                torch.save(model.state_dict(), f"checkpoints/regression/best_r2_{model_name}_model.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"checkpoints/regression/best_{model_name}_model.pt")

    return train_losses, val_losses, train_accuracies, val_accuracies
            