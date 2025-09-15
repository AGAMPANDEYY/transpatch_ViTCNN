"""
Greedy Pixel-wise Patch Optimization for PIDNet-s

This package implements a greedy pixel-wise patch optimization approach for PIDNet-s.
It creates adversarial patches that can be applied to images to cause misclassification
in the PIDNet-s semantic segmentation model.

Modules:
    - greedy_optimizer: Core implementation of the greedy patch optimization algorithm
    - main: Script to run the optimization process
    - evaluate: Script to evaluate and visualize the optimized patch on test images
    - example: Example script demonstrating the complete workflow
"""

from greedy_patch.greedy_optimizer import GreedyPatchOptimizer
from greedy_patch.evaluate import evaluate_patch, visualize_results

__all__ = ['GreedyPatchOptimizer', 'evaluate_patch', 'visualize_results']