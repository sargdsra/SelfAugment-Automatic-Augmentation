"""
Rotation prediction evaluation for self-supervised representations.

Implements the linear evaluation task described in Section 3.1:
- Input image undergoes one of 4 rotations {0°, 90°, 180°, 270°}
- A linear classifier predicts the rotation angle
- Accuracy is used as the evaluation metric

Key finding from paper: Rotation prediction has rank correlation > 0.94
with supervised performance across multiple datasets and tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm


class RotationEvaluator:
    """
    Linear evaluation of representations using rotation prediction.
    
    Trains a linear classifier on top of frozen representations
    to predict the rotation angle {0°, 90°, 180°, 270°}.
    
    References:
    - Gidaris et al. (2018): Unsupervised representation learning by predicting image rotations
    - Kolesnikov et al. (2019): Revisiting self-supervised visual representation learning
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 input_dim: int,
                 num_rotations: int = 4,
                 device: str = 'cuda',
                 batch_size: int = 128,
                 learning_rate: float = 0.01,
                 epochs: int = 50):
        """
        Initialize rotation evaluator.
        
        Args:
            backbone: Pre-trained feature extractor (frozen)
            input_dim: Dimensionality of backbone output
            num_rotations: Number of rotation classes (default: 4)
            device: 'cuda' or 'cpu'
            batch_size: Batch size for training
            learning_rate: Learning rate for linear classifier
            epochs: Number of training epochs
        """
        self.backbone = backbone
        self.input_dim = input_dim
        self.num_rotations = num_rotations
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Linear classifier
        self.classifier = nn.Linear(input_dim, num_rotations).to(device)
        
    def _apply_rotation(self, x: torch.Tensor, angle_idx: int) -> torch.Tensor:
        """
        Apply rotation to batch of images.
        
        Args:
            x: Input images [B, C, H, W]
            angle_idx: 0=0°, 1=90°, 2=180°, 3=270°
            
        Returns:
            Rotated images
        """
        k = angle_idx  # Number of 90-degree rotations
        return torch.rot90(x, k, dims=[-2, -1])
    
    def prepare_rotation_data(self, 
                              images: torch.Tensor,
                              num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for rotation evaluation.
        
        Each image is rotated by 0°, 90°, 180°, 270° and labeled
        with the corresponding rotation index.
        
        Args:
            images: Input images [N, C, H, W]
            num_samples: Number of samples to use (None for all)
            
        Returns:
            Tuple of (augmented_images, labels)
        """
        if num_samples is not None:
            images = images[:num_samples]
        
        N = len(images)
        all_images = []
        all_labels = []
        
        for angle_idx in range(self.num_rotations):
            rotated = self._apply_rotation(images, angle_idx)
            labels = torch.full((N,), angle_idx, dtype=torch.long)
            all_images.append(rotated)
            all_labels.append(labels)
        
        return torch.cat(all_images, dim=0), torch.cat(all_labels, dim=0)
    
    def extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Extract features from frozen backbone.
        
        Args:
            dataloader: DataLoader providing images
            
        Returns:
            Tensor of features [N, input_dim]
        """
        self.backbone.eval()
        features = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle both (images,) and (images, labels) formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # Extract features
                feat = self.backbone(images)
                if isinstance(feat, tuple):
                    feat = feat[0]  # Some backbones return multiple outputs
                
                features.append(feat.cpu())
        
        return torch.cat(features, dim=0)
    
    def train_linear_classifier(self,
                                features: torch.Tensor,
                                labels: torch.Tensor,
                                verbose: bool = True) -> float:
        """
        Train linear classifier on frozen features.
        
        Args:
            features: Extracted features [N, input_dim]
            labels: Rotation labels [N]
            verbose: Print progress
            
        Returns:
            Final training accuracy
        """
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        
        criterion = nn.CrossEntropyLoss()
        
        self.classifier.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}', 
                       disable=not verbose)
            
            for batch_features, batch_labels in pbar:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)
                
                if verbose:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': correct / total
                    })
        
        return correct / total
    
    def evaluate(self, images: torch.Tensor, verbose: bool = True) -> float:
        """
        Full evaluation pipeline: prepare data, extract features, train classifier.
        
        Args:
            images: Input images [N, C, H, W]
            verbose: Print progress
            
        Returns:
            Rotation prediction accuracy
        """
        # Prepare rotation data
        if verbose:
            print("Preparing rotation data...")
        rot_images, rot_labels = self.prepare_rotation_data(images)
        
        # Create dataloader
        dataset = TensorDataset(rot_images)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Extract features
        if verbose:
            print("Extracting features from backbone...")
        features = self.extract_features(dataloader)
        
        # Train linear classifier
        if verbose:
            print("Training linear classifier...")
        accuracy = self.train_linear_classifier(features, rot_labels, verbose)
        
        if verbose:
            print(f"Rotation prediction accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def evaluate_on_dataloader(self, dataloader: DataLoader, 
                              num_samples: Optional[int] = None,
                              verbose: bool = True) -> float:
        """
        Evaluate on a DataLoader directly.
        
        Args:
            dataloader: DataLoader providing images
            num_samples: Number of samples to use
            verbose: Print progress
            
        Returns:
            Rotation prediction accuracy
        """
        # Collect all images
        images = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images.append(batch[0])
            else:
                images.append(batch)
            if num_samples is not None and len(images) * dataloader.batch_size >= num_samples:
                break
        
        images = torch.cat(images, dim=0)
        if num_samples is not None:
            images = images[:num_samples]
        
        return self.evaluate(images, verbose)
    
    def compute_correlation(self, 
                           supervised_scores: List[float],
                           verbose: bool = True) -> float:
        """
        Compute Spearman rank correlation with supervised scores.
        
        Args:
            supervised_scores: List of supervised performance scores
            verbose: Print result
            
        Returns:
            Spearman rank correlation coefficient
        """
        from scipy.stats import spearmanr
        
        # This would typically be called with multiple policies
        # Here we compute on the fly - in practice, you would collect
        # rotation scores for each policy and compute correlation
        
        # Placeholder - should be implemented for actual correlation analysis
        correlation, p_value = spearmanr(supervised_scores, [0.0] * len(supervised_scores))
        
        if verbose:
            print(f"Spearman rank correlation: {correlation:.4f} (p={p_value:.4f})")
        
        return correlation