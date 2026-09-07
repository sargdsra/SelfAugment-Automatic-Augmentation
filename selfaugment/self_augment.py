"""
SelfAugment: Main algorithm for automatic augmentation selection.

Implements Algorithm 1 from the paper with three key adaptations:
1. Selecting base policy via rotation evaluation
2. Searching policies using self-supervised loss functions
3. Retraining MoCo with the selected policy

Loss functions explored:
- Min evaluation error: Minimize rotation prediction loss
- Min InfoNCE: Minimize contrastive loss (easier pairs)
- Max InfoNCE: Maximize contrastive loss (harder pairs)
- Minimax: Minimize rotation loss + Maximize InfoNCE
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from sklearn.model_selection import KFold
from tqdm import tqdm
import copy

from selfaugment.augmentation_ops import OPERATIONS, MAGNITUDE_RANGES
from selfaugment.augmentation_policy import SubPolicy, AugmentationPolicy
from selfaugment.rotation_evaluator import RotationEvaluator


class SelfAugment:
    """
    SelfAugment: Fully unsupervised augmentation policy selection.
    
    Algorithm 1 from the paper:
    1. Select base policy by evaluating individual transformations
    2. Split data into K folds
    3. For each fold, train MoCo with base policy
    4. Use Bayesian optimization to search for policies
    5. Merge top policies from each fold
    6. Retrain MoCo with final policy
    
    Key insight: Rotation prediction correlates with supervised performance,
    enabling unsupervised policy selection.
    """
    
    def __init__(self,
                 operation_names: Optional[List[str]] = None,
                 k_folds: int = 5,
                 n_sub_policies: int = 10,
                 n_ops_per_policy: int = 2,
                 top_k_policies: int = 10,
                 batch_size: int = 128,
                 epochs_evaluation: int = 50,
                 epochs_pretrain: int = 100,
                 verbose: bool = True):
        """
        Initialize SelfAugment.
        
        Args:
            operation_names: List of operation names (default: all)
            k_folds: Number of folds for cross-validation (K)
            n_sub_policies: Number of sub-policies (N_tau)
            n_ops_per_policy: Number of operations per sub-policy (N_tau)
            top_k_policies: Top policies to merge from each fold (P)
            batch_size: Batch size for training
            epochs_evaluation: Epochs for linear evaluation
            epochs_pretrain: Epochs for MoCo pretraining
            verbose: Print progress
        """
        self.operation_names = operation_names or list(OPERATIONS.keys())
        self.k_folds = k_folds
        self.n_sub_policies = n_sub_policies
        self.n_ops_per_policy = n_ops_per_policy
        self.top_k_policies = top_k_policies
        self.batch_size = batch_size
        self.epochs_evaluation = epochs_evaluation
        self.epochs_pretrain = epochs_pretrain
        self.verbose = verbose
        
        # Filter operations with valid magnitude ranges
        self.operation_names = [
            op for op in self.operation_names
            if op in MAGNITUDE_RANGES and MAGNITUDE_RANGES[op][1] > 0
        ]
    
    def select_base_policy(self,
                           backbone: nn.Module,
                           images: torch.Tensor,
                           evaluator: RotationEvaluator) -> Tuple[str, AugmentationPolicy]:
        """
        Select the base augmentation policy (Algorithm 1, lines 2-5).
        
        Evaluates each individual transformation and selects the one
        with the best rotation prediction accuracy.
        
        Args:
            backbone: Feature extractor
            images: Training images
            evaluator: RotationEvaluator instance
            
        Returns:
            Tuple of (best_op_name, base_policy)
        """
        if self.verbose:
            print("Selecting base policy...")
        
        best_acc = -1
        best_op = None
        
        for op_name in self.operation_names:
            if self.verbose:
                print(f"  Evaluating {op_name}...")
            
            # Create policy with single operation
            mag_range = MAGNITUDE_RANGES.get(op_name, (0, 0))
            magnitude = (mag_range[0] + mag_range[1]) / 2  # Midpoint
            
            # Apply operation to images
            augmented = self._apply_operation_batch(images, op_name, magnitude)
            
            # Evaluate rotation accuracy
            acc = evaluator.evaluate(augmented, verbose=False)
            
            if self.verbose:
                print(f"    Rotation accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_op = op_name
        
        # Create policy with best operation
        sub_policy = SubPolicy([(best_op, best_acc, 1.0)])
        policy = AugmentationPolicy([sub_policy])
        
        if self.verbose:
            print(f"Base policy: {best_op} (acc: {best_acc:.4f})")
        
        return best_op, policy
    
    def _apply_operation_batch(self, images: torch.Tensor, 
                              op_name: str, magnitude: float) -> torch.Tensor:
        """
        Apply a single operation to a batch of images.
        
        Args:
            images: Input images [N, C, H, W]
            op_name: Operation name
            magnitude: Operation magnitude
            
        Returns:
            Augmented images
        """
        # Convert to PIL, apply, convert back
        # This is simplified - actual implementation would use proper transforms
        augmented = []
        op_fn = OPERATIONS.get(op_name)
        
        for img in images:
            # Convert tensor to PIL
            img_pil = self._tensor_to_pil(img)
            # Apply operation
            augmented_img = op_fn(img_pil, magnitude)
            # Convert back to tensor
            augmented.append(self._pil_to_tensor(augmented_img))
        
        return torch.stack(augmented)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Assume tensor is [C, H, W] normalized to [0, 1]
        import torchvision.transforms as T
        from PIL import Image
        # Denormalize
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        import torchvision.transforms as T
        return T.ToTensor()(img)
    
    def search_policies_fold(self,
                            backbone: nn.Module,
                            train_images: torch.Tensor,
                            val_images: torch.Tensor,
                            evaluator: RotationEvaluator,
                            loss_function: str = 'minimax') -> List[AugmentationPolicy]:
        """
        Search for augmentation policies on a single fold.
        
        Implements the Bayesian optimization search (Algorithm 1, lines 11-13).
        
        Args:
            backbone: Feature extractor
            train_images: Training images for MoCo
            val_images: Validation images for evaluation
            evaluator: RotationEvaluator
            loss_function: 'min_eval_error', 'min_infonce', 'max_infonce', 'minimax'
            
        Returns:
            List of top policies
        """
        if self.verbose:
            print("  Searching policies...")
        
        # This is a simplified implementation
        # The actual paper uses Bayesian optimization with Gaussian Processes
        # Here we use random search as a simpler alternative
        
        candidates = []
        
        for _ in range(50):  # Number of search iterations
            # Randomly generate a policy
            policy = AugmentationPolicy.random(
                n_sub_policies=self.n_sub_policies,
                n_ops_per_policy=self.n_ops_per_policy
            )
            
            # Apply policy to validation images
            val_augmented = []
            for img in val_images:
                augmented = policy.apply(self._tensor_to_pil(img))
                val_augmented.append(self._pil_to_tensor(augmented))
            val_augmented = torch.stack(val_augmented)
            
            # Evaluate using loss function
            if loss_function == 'min_eval_error':
                # Minimize rotation error
                loss = -evaluator.evaluate(val_augmented, verbose=False)
            elif loss_function == 'min_infonce':
                # Minimize InfoNCE (easier pairs)
                loss = self._compute_infonce(backbone, val_augmented)
            elif loss_function == 'max_infonce':
                # Maximize InfoNCE (harder pairs)
                loss = -self._compute_infonce(backbone, val_augmented)
            elif loss_function == 'minimax':
                # Minimize rotation + Maximize InfoNCE
                rot_acc = evaluator.evaluate(val_augmented, verbose=False)
                infonce = self._compute_infonce(backbone, val_augmented)
                loss = -rot_acc + infonce  # Actually minimization of loss = -rot_acc + infonce
            else:
                raise ValueError(f"Unknown loss function: {loss_function}")
            
            candidates.append((loss, policy))
        
        # Sort by loss and return top K
        candidates.sort(key=lambda x: x[0])
        top_policies = [policy for _, policy in candidates[:self.top_k_policies]]
        
        return top_policies
    
    def _compute_infonce(self, backbone: nn.Module, images: torch.Tensor) -> float:
        """
        Compute InfoNCE loss on images.
        
        This is a simplified approximation - actual implementation would
        use a full contrastive learning setup.
        
        Args:
            backbone: Feature extractor
            images: Input images
            
        Returns:
            InfoNCE loss value
        """
        # Simplified InfoNCE computation
        # In practice, this would involve:
        # 1. Two augmentations of each image
        # 2. Projection head
        # 3. Temperature scaling
        # 4. Negative sampling
        
        # Placeholder - implement actual InfoNCE in full implementation
        with torch.no_grad():
            features = backbone(images)
            if isinstance(features, tuple):
                features = features[0]
            
            # Random projection for demo
            projection = torch.randn(features.shape[1], 128).to(features.device)
            z = features @ projection
            
            # Compute similarity
            sim = z @ z.T
            # InfoNCE (simplified)
            loss = -torch.log(torch.exp(sim.mean()) / torch.exp(sim).sum())
        
        return loss.item()
    
    def merge_policies(self, fold_policies: List[List[AugmentationPolicy]]) -> AugmentationPolicy:
        """
        Merge policies from all folds.
        
        Algorithm 1, line 13:
        Merge policies via T*(k) ← T*(k) ∪ Tt
        
        Args:
            fold_policies: List of policies from each fold
            
        Returns:
            Merged policy
        """
        if self.verbose:
            print("Merging policies from all folds...")
        
        all_sub_policies = []
        
        for policies in fold_policies:
            for policy in policies:
                all_sub_policies.extend(policy.sub_policies)
        
        # Select top N_sub_policies unique sub-policies
        # Remove duplicates by converting to hashable form
        unique_policies = []
        seen = set()
        
        for sp in all_sub_policies:
            # Create a hashable representation
            key = tuple([(op, round(mag, 2), round(prob, 2)) 
                        for op, mag, prob in sp.operations])
            if key not in seen:
                seen.add(key)
                unique_policies.append(sp)
        
        # Select top N_sub_policies
        selected = unique_policies[:self.n_sub_policies]
        
        return AugmentationPolicy(selected)
    
    def run(self,
            backbone_class: Callable,
            images: torch.Tensor,
            loss_function: str = 'minimax',
            verbose: bool = None) -> Tuple[AugmentationPolicy, Dict[str, Any]]:
        """
        Run the full SelfAugment algorithm.
        
        Args:
            backbone_class: Class to instantiate backbone
            images: Training images [N, C, H, W]
            loss_function: Loss function for policy selection
            verbose: Override verbose setting
            
        Returns:
            Tuple of (selected_policy, results_dict)
        """
        if verbose is not None:
            self.verbose = verbose
        
        results = {
            'base_policy': None,
            'fold_policies': [],
            'final_policy': None,
            'rotation_accuracies': []
        }
        
        # Create K-fold splits
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_policies = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images)):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Fold {fold_idx + 1}/{self.k_folds}")
                print(f"{'='*50}")
            
            train_imgs = images[train_idx]
            val_imgs = images[val_idx]
            
            # Initialize backbone
            backbone = backbone_class()
            input_dim = backbone.output_dim
            
            # Create evaluator
            evaluator = RotationEvaluator(
                backbone=backbone,
                input_dim=input_dim,
                batch_size=self.batch_size,
                epochs=self.epochs_evaluation
            )
            
            # Select base policy (Algorithm 1, lines 2-5)
            if fold_idx == 0:  # Only for first fold
                base_op, base_policy = self.select_base_policy(
                    backbone, train_imgs, evaluator
                )
                results['base_policy'] = base_op
            else:
                base_policy = results['base_policy']
            
            # Train MoCo with base policy (Algorithm 1, line 8)
            # This is a placeholder - actual training would be more complex
            backbone = self._train_moco(backbone, train_imgs, base_policy)
            
            # Search for policies (Algorithm 1, lines 10-12)
            fold_policies_found = self.search_policies_fold(
                backbone, train_imgs, val_imgs, evaluator, loss_function
            )
            
            fold_policies.append(fold_policies_found)
            results['fold_policies'].append(fold_policies_found)
            
            # Evaluate rotation accuracy (Algorithm 1, line 9)
            val_augmented = self._apply_policy_batch(val_imgs, base_policy)
            acc = evaluator.evaluate(val_augmented, verbose=False)
            results['rotation_accuracies'].append(acc)
            
            if self.verbose:
                print(f"Fold {fold_idx + 1} rotation accuracy: {acc:.4f}")
        
        # Merge policies (Algorithm 1, line 13)
        final_policy = self.merge_policies(fold_policies)
        results['final_policy'] = final_policy
        
        if self.verbose:
            print(f"\n{'='*50}")
            print("SelfAugment complete!")
            print(f"Number of sub-policies: {len(final_policy.sub_policies)}")
            print(f"Average rotation accuracy: {np.mean(results['rotation_accuracies']):.4f}")
            print(f"{'='*50}")
        
        return final_policy, results
    
    def _train_moco(self, backbone: nn.Module, images: torch.Tensor, 
                   policy: AugmentationPolicy) -> nn.Module:
        """
        Train MoCo with a given policy.
        
        This is a placeholder - actual MoCo training would be:
        1. Two augmentations per image
        2. Encoder + Projection head
        3. Momentum update
        4. Queue for negatives
        
        Returns:
            Trained backbone
        """
        # Placeholder: just return backbone
        # In actual implementation, this would run full MoCo training
        return backbone
    
    def _apply_policy_batch(self, images: torch.Tensor, 
                           policy: AugmentationPolicy) -> torch.Tensor:
        """
        Apply augmentation policy to a batch of images.
        
        Args:
            images: Input images [N, C, H, W]
            policy: AugmentationPolicy to apply
            
        Returns:
            Augmented images
        """
        augmented = []
        for img in images:
            img_pil = self._tensor_to_pil(img)
            augmented_img = policy.apply(img_pil)
            augmented.append(self._pil_to_tensor(augmented_img))
        return torch.stack(augmented)