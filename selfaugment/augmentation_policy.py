"""
Augmentation policy and sub-policy definitions.

Defines the structure for augmentation policies:
- Sub-policy: Sequence of N transformations applied sequentially
- Policy: Collection of N_sub sub-policies sampled at training time
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import random
from selfaugment.augmentation_ops import OPERATIONS, MAGNITUDE_RANGES


class SubPolicy:
    """
    A sub-policy consisting of sequential augmentation operations.
    
    Each operation has:
    - Operation name (e.g., 'rotate', 'color')
    - Magnitude (lambda): strength of the transformation
    - Probability (p): chance of applying the operation
    
    From the paper:
    A sub-policy τ is defined as the sequential application of
    Nτ consecutive transformations.
    """
    
    def __init__(self, operations: List[Tuple[str, float, float]]):
        """
        Initialize a sub-policy.
        
        Args:
            operations: List of (op_name, magnitude, probability) tuples
        """
        self.operations = operations
        
    def apply(self, img: Image.Image) -> Image.Image:
        """
        Apply all operations in sequence.
        
        Args:
            img: PIL Image
            
        Returns:
            Transformed image
        """
        x = img
        for op_name, magnitude, prob in self.operations:
            # Apply with probability p
            if random.random() < prob:
                op_fn = OPERATIONS.get(op_name)
                if op_fn:
                    x = op_fn(x, magnitude)
        return x
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sub-policy to dictionary representation."""
        return {
            'operations': [
                {'name': op_name, 'magnitude': mag, 'probability': prob}
                for op_name, mag, prob in self.operations
            ]
        }
    
    @classmethod
    def random(cls, n_ops: int = 2, op_names: Optional[List[str]] = None) -> 'SubPolicy':
        """
        Create a random sub-policy.
        
        Args:
            n_ops: Number of operations in sub-policy
            op_names: List of allowed operation names (default: all)
            
        Returns:
            Random SubPolicy
        """
        if op_names is None:
            op_names = list(OPERATIONS.keys())
        
        operations = []
        for _ in range(n_ops):
            op_name = np.random.choice(op_names)
            mag_range = MAGNITUDE_RANGES.get(op_name, (0, 0))
            magnitude = np.random.uniform(mag_range[0], mag_range[1])
            probability = np.random.uniform(0, 1)
            operations.append((op_name, magnitude, probability))
        
        return cls(operations)


class AugmentationPolicy:
    """
    Collection of sub-policies.
    
    A full policy T is a collection of N_tau sub-policies.
    At training time, one sub-policy is randomly selected for each image.
    """
    
    def __init__(self, sub_policies: List[SubPolicy]):
        """
        Initialize augmentation policy.
        
        Args:
            sub_policies: List of SubPolicy objects
        """
        self.sub_policies = sub_policies
        
    def apply(self, img: Image.Image) -> Image.Image:
        """
        Apply a randomly selected sub-policy to the image.
        
        Args:
            img: PIL Image
            
        Returns:
            Augmented image
        """
        # Select random sub-policy
        sub_policy = np.random.choice(self.sub_policies)
        return sub_policy.apply(img)
    
    def apply_all(self, img: Image.Image) -> List[Image.Image]:
        """
        Apply all sub-policies to the image.
        
        Useful for visualization and analysis.
        
        Args:
            img: PIL Image
            
        Returns:
            List of augmented images (one per sub-policy)
        """
        return [sp.apply(img) for sp in self.sub_policies]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary representation."""
        return {
            'sub_policies': [sp.to_dict() for sp in self.sub_policies]
        }
    
    @classmethod
    def random(cls, n_sub_policies: int = 10, n_ops_per_policy: int = 2) -> 'AugmentationPolicy':
        """
        Create a random augmentation policy.
        
        Args:
            n_sub_policies: Number of sub-policies
            n_ops_per_policy: Number of operations per sub-policy
            
        Returns:
            Random AugmentationPolicy
        """
        sub_policies = [
            SubPolicy.random(n_ops_per_policy)
            for _ in range(n_sub_policies)
        ]
        return cls(sub_policies)
    
    @classmethod
    def from_config(cls, config: List[List[Tuple[str, float, float]]]) -> 'AugmentationPolicy':
        """
        Create policy from configuration.
        
        Args:
            config: List of sub-policies, each a list of (op, mag, prob)
            
        Returns:
            AugmentationPolicy
        """
        sub_policies = [SubPolicy(ops) for ops in config]
        return cls(sub_policies)
    
    @classmethod
    def get_moco_v2_policy(cls) -> 'AugmentationPolicy':
        """
        Return the MoCoV2 augmentation policy (baseline from paper).
        
        Returns:
            MoCoV2 policy
        """
        # MoCoV2 uses random resized crop + random color jitter + random grayscale
        # This is a simplified representation
        # In practice, this would be implemented with standard transforms
        config = [
            [('identity', 0, 0.0)]  # Placeholder - actual MoCoV2 has more complex transforms
        ]
        return cls.from_config(config)