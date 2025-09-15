#!/usr/bin/env python
"""
Example script demonstrating how to use the greedy pixel-wise patch optimization.
This script performs a complete workflow:
1. Optimize a patch using the greedy approach
2. Evaluate the optimized patch on test images
3. Visualize the results
"""

import os
import sys
import argparse
import torch
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from greedy_patch.greedy_optimizer import GreedyPatchOptimizer
from greedy_patch.evaluate import evaluate_patch
from utils.utils import get_config_from_yaml

def setup_logger():
    """Set up a basic logger for the example script"""
    logger = logging.getLogger('greedy_patch_example')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

def main():
    parser = argparse.ArgumentParser(description='Example of greedy pixel-wise patch optimization')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--output_dir', type=str, default='greedy_patch/example_results',
                        help='Directory to save results')
    parser.add_argument('--num_images', type=int, default=3,
                        help='Number of images to use for optimization')
    parser.add_argument('--num_eval_samples', type=int, default=5,
                        help='Number of test samples to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--patch_ext', type=str, default='pt',
                        help='File extension for patch files (pt or p)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    optimization_dir = os.path.join(args.output_dir, 'optimization')
    evaluation_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(optimization_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger()
    logger.info("Starting greedy patch optimization example")
    
    # Load configuration
    config = get_config_from_yaml(args.config)
    
    # Override device if specified
    if args.device:
        config.experiment.device = args.device
    
    # Step 1: Optimize a patch using the greedy approach
    logger.info("Step 1: Optimizing patch")
    optimizer = GreedyPatchOptimizer(config, logger)
    
    patch_save_path = os.path.join(optimization_dir, f'optimized_patch.{args.patch_ext}')
    patch, iou_metrics = optimizer.optimize_patch(
        initial_patch=None,  # Start with a random patch
        num_images=args.num_images,
        save_path=patch_save_path
    )
    
    logger.info(f"Patch optimization complete. Patch saved to {patch_save_path}")
    
    # Step 2: Evaluate the optimized patch on test images
    logger.info("Step 2: Evaluating optimized patch")
    results = evaluate_patch(
        config=config,
        patch_path=patch_save_path,
        num_samples=args.num_eval_samples,
        output_dir=evaluation_dir
    )
    
    # Step 3: Print summary of results
    logger.info("Step 3: Summary of results")
    logger.info("-" * 80)
    logger.info("Optimization completed with:")
    logger.info(f"- Number of images used: {args.num_images}")
    logger.info(f"- Final patch shape: {patch.shape}")
    
    logger.info("\nEvaluation results:")
    logger.info(f"- Clean Image Mean IoU: {results['clean_mIoU']:.4f}")
    logger.info(f"- Patched Image Mean IoU: {results['patched_mIoU']:.4f}")
    logger.info(f"- IoU Reduction: {results['iou_reduction']:.4f}")
    logger.info(f"- Misclassification Rate: {results['misclassification_rate']:.4f}")
    
    logger.info("\nResults saved to:")
    logger.info(f"- Optimization results: {optimization_dir}")
    logger.info(f"- Evaluation results: {evaluation_dir}")
    logger.info("-" * 80)
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main()