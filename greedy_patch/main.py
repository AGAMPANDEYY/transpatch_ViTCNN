import sys
import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimizer
from greedy_patch.greedy_optimizer import GreedyPatchOptimizer

# Import config utilities
from utils.utils import get_config_from_yaml

def setup_logger(log_dir):
    """
    Set up a logger that writes to both console and file.
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        A configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"greedy_patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger('greedy_patch')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_results(patch, iou_metrics, save_dir, patch_ext='pt'):
    """
    Save the optimized patch and plot IoU metrics.
    
    Args:
        patch: The optimized patch tensor
        iou_metrics: Array of IoU metrics
        save_dir: Directory to save results
        patch_ext: File extension for the patch file (default: 'pt')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save patch as tensor
    torch.save(patch, os.path.join(save_dir, f'optimized_patch.{patch_ext}'))
    
    # Save patch as image
    patch_np = patch.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    plt.figure(figsize=(8, 8))
    plt.imshow(patch_np)
    plt.title('Optimized Patch')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'optimized_patch.png'))
    
    # Plot IoU metrics if available
    if len(iou_metrics) > 0:
        plt.figure(figsize=(10, 6))
        
        # Extract mean IoU values
        mean_ious = [metrics[1] for metrics in iou_metrics]  # Assuming mIoU is the second value
        
        plt.plot(range(1, len(mean_ious) + 1), mean_ious, marker='o')
        plt.xlabel('Image Number')
        plt.ylabel('Mean IoU')
        plt.title('Mean IoU Progression During Optimization')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'miou_progression.png'))
        # Save raw metrics data
        np.save(os.path.join(save_dir, 'iou_metrics.npy'), iou_metrics)

def main():
    """
    Main function to run the greedy patch optimization.
    """
    parser = argparse.ArgumentParser(description='Greedy Pixel-wise Patch Optimization for PIDNet-s')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--initial_patch', type=str, default=None,
                        help='Path to an initial patch to start optimization from')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to use for optimization')
    parser.add_argument('--output_dir', type=str, default='greedy_patch/results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--patch_ext', type=str, default='pt',
                        help='File extension for patch files (pt or p)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)# Set up logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Starting greedy patch optimization with config: {args.config}")
    
    # Load configuration
    config = get_config_from_yaml(args.config)
    
    # Override device if specified
    if args.device:
        config.experiment.device = args.device
    
    # Initialize optimizer
    optimizer = GreedyPatchOptimizer(config, logger)
    
    # Run optimization
    save_path = os.path.join(args.output_dir, f'optimized_patch.{args.patch_ext}')
    patch, iou_metrics = optimizer.optimize_patch(
        initial_patch=args.initial_patch,
        num_images=args.num_images,
        save_path=save_path
    )
    
    # Save and visualize results
    save_results(patch, iou_metrics, args.output_dir, args.patch_ext)
    
    logger.info(f"Optimization complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()