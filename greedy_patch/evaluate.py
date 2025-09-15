import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from pretrained_models.models import Models
from dataset.cityscapes import Cityscapes
from metrics.performance import SegmentationMetric
from patch.create import Patch
from utils.utils import get_config_from_yaml

def setup_logger():
    """Set up a basic logger for evaluation output"""
    logger = logging.getLogger('evaluate')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

def visualize_results(image, true_label, clean_pred, patched_image, patched_pred, save_path=None):
    """
    Visualize the original image, patched image, and their predictions.
    
    Args:
        image: Original image tensor
        true_label: Ground truth label tensor
        clean_pred: Prediction on clean image
        patched_image: Image with patch applied
        patched_pred: Prediction on patched image
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy arrays for visualization
    image_np = image[0].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
    true_label_np = true_label[0].cpu().numpy()
    clean_pred_np = clean_pred[0].cpu().numpy()
    patched_image_np = patched_image[0].cpu().permute(1, 2, 0).numpy()
    patched_pred_np = patched_pred[0].cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot ground truth label
    axes[0, 1].imshow(true_label_np, cmap='tab20')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Plot clean prediction
    axes[0, 2].imshow(clean_pred_np, cmap='tab20')
    axes[0, 2].set_title('Clean Prediction')
    axes[0, 2].axis('off')
    
    # Plot patched image
    axes[1, 0].imshow(patched_image_np)
    axes[1, 0].set_title('Patched Image')
    axes[1, 0].axis('off')
    
    # Plot patched prediction
    axes[1, 1].imshow(patched_pred_np, cmap='tab20')
    axes[1, 1].set_title('Patched Prediction')
    axes[1, 1].axis('off')
    
    # Plot difference between clean and patched predictions
    diff = (clean_pred_np != patched_pred_np).astype(np.float32)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Prediction Differences')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_patch(config, patch_path, num_samples=10, output_dir='greedy_patch/evaluation'):
    """
    Evaluate a patch on test images.
    
    Args:
        config: Configuration object
        patch_path: Path to the patch file
        num_samples: Number of test samples to evaluate
        output_dir: Directory to save evaluation results
    """
    logger = setup_logger()
    logger.info(f"Evaluating patch: {patch_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(config.experiment.device)
    
    # Load patch
    patch = torch.load(patch_path, map_location=device)
    logger.info(f"Loaded patch with shape: {patch.shape}")
    
    # Initialize model
    model = Models(config)
    model.get()
    logger.info("Model loaded successfully")
    
    # Initialize patch application function
    apply_patch = Patch(config).apply_patch
    
    # Initialize metrics
    clean_metric = SegmentationMetric(config)
    patched_metric = SegmentationMetric(config)
    
    # Load test dataset
    test_dataset = Cityscapes(
        root=config.dataset.root,
        list_path=config.dataset.val,
        num_classes=config.dataset.num_classes,
        multi_scale=False,
        flip=False,
        ignore_label=config.train.ignore_label,
        base_size=config.test.base_size,
        crop_size=(config.test.height, config.test.width),
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,  # Process one image at a time for visualization
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=config.test.pin_memory,
    )
    
    # Evaluate on test samples
    logger.info(f"Evaluating on {num_samples} test samples")
    
    # Track misclassification rates
    total_pixels = 0
    misclassified_pixels = 0
    
    for i, batch in enumerate(tqdm(test_dataloader)):
        if i >= num_samples:
            break
            
        image, true_label, _, _, _ = batch
        image, true_label = image.to(device), true_label.to(device)
        
        # Forward pass on clean image
        with torch.no_grad():
            clean_output = model.predict(image, true_label.shape)
            clean_pred = clean_output.argmax(dim=1)
            
            # Apply patch to image
            patched_image, patched_label = apply_patch(image, true_label, patch)
            
            # Forward pass on patched image
            patched_output = model.predict(patched_image, patched_label.shape)
            patched_pred = patched_output.argmax(dim=1)
            
            # Update metrics
            clean_metric.update(clean_output, true_label)
            patched_metric.update(patched_output, patched_label)
            
            # Calculate misclassification
            valid_mask = patched_label != config.train.ignore_label
            total_valid = valid_mask.sum().item()
            misclassified = ((patched_pred != patched_label) & valid_mask).sum().item()
            
            total_pixels += total_valid
            misclassified_pixels += misclassified
            
            # Visualize results
            save_path = os.path.join(output_dir, f"sample_{i}.png")
            visualize_results(
                image, true_label, clean_pred, 
                patched_image, patched_pred, 
                save_path=save_path
            )
    
    # Calculate metrics
    clean_pixAcc, clean_mIoU = clean_metric.get()
    patched_pixAcc, patched_mIoU = patched_metric.get()
    
    # Calculate misclassification rate
    misclassification_rate = misclassified_pixels / total_pixels if total_pixels > 0 else 0
    
    # Print results
    logger.info("-" * 80)
    logger.info("Evaluation Results:")
    logger.info(f"Clean Image - Pixel Accuracy: {clean_pixAcc:.4f}, Mean IoU: {clean_mIoU:.4f}")
    logger.info(f"Patched Image - Pixel Accuracy: {patched_pixAcc:.4f}, Mean IoU: {patched_mIoU:.4f}")
    logger.info(f"Misclassification Rate: {misclassification_rate:.4f}")
    logger.info(f"IoU Reduction: {clean_mIoU - patched_mIoU:.4f}")
    logger.info("-" * 80)
    
    # Save metrics to file
    results = {
        'clean_pixAcc': clean_pixAcc,
        'clean_mIoU': clean_mIoU,
        'patched_pixAcc': patched_pixAcc,
        'patched_mIoU': patched_mIoU,
        'misclassification_rate': misclassification_rate,
        'iou_reduction': clean_mIoU - patched_mIoU
    }
    
    np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)
    
    # Also save as text
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate an optimized patch on test images')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--patch', type=str, required=True,
                        help='Path to the patch file to evaluate')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='greedy_patch/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config_from_yaml(args.config)
    
    # Override device if specified
    if args.device:
        config.experiment.device = args.device
    
    # Evaluate patch
    evaluate_patch(
        config=config,
        patch_path=args.patch,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()