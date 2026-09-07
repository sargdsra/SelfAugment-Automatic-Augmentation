# SelfAugment: Automatic Augmentation Policies for Self-Supervised Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

## 📖 About

Implementation of the paper:

**"SelfAugment: Automatic Augmentation Policies for Self-Supervised Learning"**  
by Colorado J Reed, Sean Metzger, Aravind Srinivas, Trevor Darrell, Kurt Keutzer (2021)

This package provides fully unsupervised augmentation policy selection for self-supervised learning.

## 🎯 Key Features

- **Self-Supervised Evaluation**: Uses rotation prediction to evaluate representations without labels
- **SelfRandAugment**: Sampling-based augmentation selection
- **SelfAugment**: Search-based augmentation selection with Bayesian optimization
- **High Correlation**: Rotation prediction has rank correlation > 0.94 with supervised performance
- **Fully Unsupervised**: No labels needed for augmentation selection

## 🔧 Installation

```bash
git clone https://github.com/sargdsra/SelfAugment-Automatic-Augmentation.git
cd SelfAugment-Automatic-Augmentation
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start
```python
import torch
from selfaugment import SelfAugment, SelfRandAugment
from selfaugment.rotation_evaluator import RotationEvaluator

# Load your unlabeled images
images = torch.randn(1000, 3, 32, 32)  # Replace with actual data

# SelfRandAugment: Grid search over (N, lambda)
searcher = SelfRandAugment(
    n_values=[1, 2, 3],
    lambda_values=[4, 7, 11],
    num_sub_policies=5
)
results = searcher.search(backbone, images)
best_policy = results['best_policy']

# SelfAugment: Full algorithm with K-fold search
augmenter = SelfAugment(
    k_folds=5,
    n_sub_policies=10,
    n_ops_per_policy=2,
    top_k_policies=10
)
best_policy, results = augmenter.run(backbone, images)
```

## 📚 References

- Reed, C. J., Metzger, S., Srinivas, A., Darrell, T., & Keutzer, K. (2021). SelfAugment: Automatic Augmentation Policies for Self-Supervised Learning.

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations.

- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning.

