"""
SelfRandAugment: Sampling-based augmentation selection.

Adaptation of RandAugment for self-supervised learning.

RandAugment simplifies augmentation search by:
1. Using a single shared magnitude (lambda) for all operations
2. Applying the same number of operations (N) per sub-policy
3. Using uniform probability for all operations

SelfRandAugment uses rotation evaluation to select the best (N, lambda) combination.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import itertools
from selfaugment.augmentation_ops import OPERATIONS, MAGNITUDE_RANGES
from selfaugment.augmentation_policy import SubPolicy, AugmentationPolicy
from selfaugment.rotation_evaluator import RotationEvaluator


class SelfRandAugment:
    """
    Self-supervised RandAugment implementation.
    
    Searches over (N_tau, lambda) combinations using rotation evaluation
    to select the best augmentation policy.
    
    From the paper:
    "RandAugment makes the following simplifying assumptions:
    (i) all transformations share a single, discrete magnitude λ ∈ [1,30]
    (ii) all sub-policies apply the same number of transformations N_tau
    (iii) all transformations are applied with uniform probability p = K_T^{-1}"
    """
    
    def __init__(self,
                 operation_names: Optional[List[str]] = None,
                 n_values: List[int] = [1, 2, 3],
                 lambda_values: List[int] = [4, 5, 7, 9, 11, 13],
                 num_sub_policies: int = 5,
                 batch_size: int = 128,
                 epochs: int = 50):
        """
        Initialize SelfRandAugment.
        
        Args:
            operation_names: List of operation names to use (default: all)
            n_values: List of N (number of operations per sub-policy)
            lambda_values: List of lambda (magnitude) values to search
            num_sub_policies: Number of sub-policies in the final policy
            batch_size: Batch size for rotation evaluation
            epochs: Epochs for linear classifier training
        """
        self.operation_names = operation_names or list(OPERATIONS.keys())
        self.n_values = n_values
        self.lambda_values = lambda_values
        self.num_sub_policies = num_sub_policies
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Remove operations without valid magnitude ranges
        self.operation_names = [
            op for op in self.operation_names
            if op in MAGNITUDE_RANGES and MAGNITUDE_RANGES[op][1] > 0
        ]
        
    def create_policy_from_params(self, n: int, lam: int) -> AugmentationPolicy:
        """
        Create a RandAugment policy from parameters.
        
        Args:
            n: Number of operations per sub-policy
            lam: Shared magnitude for all operations
            
        Returns:
            AugmentationPolicy
        """
        sub_policies = []
        
        for _ in range(self.num_sub_policies):
            # Randomly select n operations
            ops = np.random.choice(self.operation_names, n, replace=False)
            
            # Create sub-policy with uniform probability
            operations = []
            for op_name in ops:
                # Apply with uniform probability
                prob = 1.0 / len(self.operation_names)
                # Scale magnitude based on operation range
                mag_range = MAGNITUDE_RANGES.get(op_name, (0, 0))
                # Use lambda as fraction of max magnitude
                magnitude = (lam / 30.0) * mag_range[1]
                operations.append((op_name, magnitude, prob))
            
            sub_policies.append(SubPolicy(operations))
        
        return AugmentationPolicy(sub_policies)
    
    def search(self,
               backbone: nn.Module,
               images: torch.Tensor,
               evaluator: Optional[RotationEvaluator] = None,
               verbose: bool = True) -> Dict[str, Any]:
        """
        Search for best (N, lambda) combination using rotation evaluation.
        
        Args:
            backbone: Pre-trained feature extractor
            images: Image dataset for evaluation
            evaluator: Optional RotationEvaluator (will create if None)
            verbose: Print progress
            
        Returns:
            Dictionary with best parameters and results
        """
        if evaluator is None:
            input_dim = backbone.output_dim  # Assumes backbone has output_dim
            evaluator = RotationEvaluator(
                backbone=backbone,
                input_dim=input_dim,
                batch_size=self.batch_size,
                epochs=self.epochs
            )
        
        best_accuracy = -1
        best_params = None
        results = []
        
        # Grid search over (N, lambda)
        for n, lam in itertools.product(self.n_values, self.lambda_values):
            if verbose:
                print(f"\nEvaluating (N={n}, lambda={lam})...")
            
            # Create policy
            policy = self.create_policy_from_params(n, lam)
            
            # Apply policy to images
            augmented_images = []
            for img in images:
                # Convert to PIL, apply, convert back
                # This is simplified - actual implementation would use
                # appropriate data loading pipeline
                augmented = policy.apply(img)
                augmented_images.append(augmented)
            
            augmented_images = torch.stack(augmented_images)
            
            # Evaluate using rotation prediction
            accuracy = evaluator.evaluate(augmented_images, verbose=False)
            
            if verbose:
                print(f"Rotation accuracy: {accuracy:.4f}")
            
            results.append({
                'n': n,
                'lambda': lam,
                'rotation_accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (n, lam)
        
        # Create best policy
        best_n, best_lam = best_params
        best_policy = self.create_policy_from_params(best_n, best_lam)
        
        return {
            'best_n': best_n,
            'best_lambda': best_lam,
            'best_accuracy': best_accuracy,
            'all_results': results,
            'best_policy': best_policy
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SelfRandAugment':
        """
        Create SelfRandAugment from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SelfRandAugment instance
        """
        return cls(
            operation_names=config.get('operation_names', None),
            n_values=config.get('n_values', [1, 2, 3]),
            lambda_values=config.get('lambda_values', [4, 5, 7, 9, 11, 13]),
            num_sub_policies=config.get('num_sub_policies', 5),
            batch_size=config.get('batch_size', 128),
            epochs=config.get('epochs', 50)
        )