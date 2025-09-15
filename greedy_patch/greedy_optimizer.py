import sys
import pickle
# Save the original sys.path
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")
from dataset.cityscapes import Cityscapes

from pretrained_models.models import Models
from pretrained_models.PIDNet.model import PIDNet, get_pred_model

from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch

import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# Restore original sys.path to avoid conflicts or shadowing
sys.path = original_sys_path

class GreedyPatchOptimizer:
    def __init__(self, config, main_logger):
        """
        Initialize the greedy patch optimizer.
        
        Args:
            config: Configuration object with parameters
            main_logger: Logger for printing information
        """
        self.config = config
        self.device = config.experiment.device
        self.logger = main_logger
        self.patch_size = config.patch.size
        self.apply_patch = Patch(config).apply_patch
        
        # Dataset setup
        cityscape_train = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.train,
            num_classes=config.dataset.num_classes,
            multi_scale=config.train.multi_scale,
            flip=config.train.flip,
            ignore_label=config.train.ignore_label,
            base_size=config.train.base_size,
            crop_size=(config.train.height, config.train.width),
            scale_factor=config.train.scale_factor
        )
        
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=cityscape_train,
            batch_size=1,  # We process one image at a time for greedy optimization
            shuffle=config.train.shuffle,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            drop_last=config.train.drop_last
        )
        
        # Model setup - specifically load PIDNet-s
        self.model = Models(self.config)
        self.model.get()
        
        # Loss function
        self.criterion = PatchLoss(self.config)
        
        # Metrics
        self.metric = SegmentationMetric(config)
        
        # Patch adjustment parameters
        self.delta = 2/255  # The adjustment step size (+2/255, 0, -2/255)
        
  

    def load_patch(self, patch_path=None):
            """
            Load an existing patch or create a new random one if path is None.
        
            Args:
                patch_path: Path to the existing patch file
        
            Returns:
                The loaded or newly created patch
            """
            if patch_path:
                # First try the exact path
                if os.path.exists(patch_path):
                    self.logger.info(f"Loading patch from {patch_path}")
                    try:
                        # Try torch.load
                        patch = torch.load(patch_path, map_location=self.device, weights_only=False)
                    except Exception as e:
                        self.logger.warning(f"torch.load failed: {e}, trying pickle.load...")
                        patch = pickle.load(open(patch_path, "rb"))[0]  # Your patches are pickled lists
                    return patch.to(self.device)
        
                # Try alternate extensions
                base_path, ext = os.path.splitext(patch_path)
                for try_ext in ['.pt', '.p']:
                    try_path = base_path + try_ext
                    if os.path.exists(try_path):
                        self.logger.info(f"Loading patch from {try_path}")
                        try:
                            patch = torch.load(try_path, map_location=self.device, weights_only=False)
                        except Exception as e:
                            self.logger.warning(f"torch.load failed: {e}, trying pickle.load...")
                            patch = pickle.load(open(try_path, "rb"))[0]
                        return patch.to(self.device)
        
                self.logger.info(f"Could not find patch at {patch_path} with .pt or .p")
                self.logger.info("Creating new random patch")
            else:
                self.logger.info("No patch path provided. Creating new random patch")
        
            # Create a new random patch if no file found or no path provided
            patch = torch.rand((3, self.patch_size, self.patch_size), device=self.device)
            return patch

    def compute_priority_map(self, image, patch, true_label):
        """
        Compute a pixel-level priority map based on gradient values.
        
        Args:
            image: The input image
            patch: The current patch
            true_label: The ground truth label
            
        Returns:
            A priority map with values for each pixel in the patch
        """
        # Create a copy of the patch that requires gradients
        patch_with_grad = patch.clone().detach().requires_grad_(True)
        
        # Apply the patch to the image
        patched_image, patched_label = self.apply_patch(image, true_label, patch_with_grad)
        
        # Ensure label is the correct dtype for loss computation
        patched_label = patched_label.long()

        # Forward pass
        output = self.model.predict(patched_image, patched_label.shape)
        
        # Compute loss
        with torch.no_grad():
            clean_output = self.model.predict(image, patched_label.shape)
            
        # Compute loss (using the same loss as in the reference implementation)
        pred_labels = output.argmax(dim=1)
        correct_pixels = (pred_labels == patched_label) & (patched_label != self.config.train.ignore_label)
        num_correct = correct_pixels.sum().item()
        
        if num_correct > 0:
            loss = self.criterion.compute_loss_transegpgd_stage1(output, patched_label, clean_output)
        else:
            loss = self.criterion.compute_loss_transegpgd_stage2(output, patched_label, clean_output)
        
        # Compute gradients
        loss.backward()
        
        # Get the gradients
        patch_grad = patch_with_grad.grad
        
        # Compute priority based on sum of absolute gradient values across channels
        priority_map = torch.sum(torch.abs(patch_grad), dim=0)
        
        return priority_map
    
    def evaluate_patch(self, image, patch, true_label):
        """
        Evaluate the effectiveness of a patch by measuring misclassification.
        
        Args:
            image: The input image
            patch: The patch to evaluate
            true_label: The ground truth label
            
        Returns:
            A score representing the effectiveness of the patch (higher is better)
        """
        with torch.no_grad():
            # Apply the patch to the image
            patched_image, patched_label = self.apply_patch(image, true_label, patch)
            
            # Forward pass
            output = self.model.predict(patched_image, patched_label.shape)
            
            # Compute metrics
            pred_labels = output.argmax(dim=1)
            correct_pixels = (pred_labels == patched_label) & (patched_label != self.config.train.ignore_label)
            num_correct = correct_pixels.sum().item()
            
            # Lower number of correct pixels means better patch
            return -num_correct
    
    def optimize_patch(self, initial_patch=None, num_images=10, save_path=None):
        """
        Perform greedy pixel-wise patch optimization.
        
        Args:
            initial_patch: Initial patch to start with (if None, a random one will be created)
            num_images: Number of images to use for optimization
            save_path: Path to save the optimized patch
            
        Returns:
            The optimized patch and IoU metrics
        """
        # Load or create initial patch
        patch = self.load_patch(initial_patch)
        
        start_time = time.time()
        self.logger.info("Starting greedy pixel-wise patch optimization")
        
        # Keep track of IoU metrics
        IoU = []
        
        # Process a subset of images
        for i_iter, batch in enumerate(self.train_dataloader):
            if i_iter >= num_images:
                break
                
            image, true_label, _, _, _ = batch
            image, true_label = image.to(self.device), true_label.to(self.device)
            
            self.logger.info(f"Processing image {i_iter+1}/{num_images}")
            
            # Compute priority map for this image
            priority_map = self.compute_priority_map(image, patch, true_label)
            
            # Flatten the priority map and get indices sorted by priority (highest to lowest)
            flat_priorities = priority_map.view(-1)
            sorted_indices = torch.argsort(flat_priorities, descending=True)
            
            # Convert flat indices to 2D coordinates
            h, w = priority_map.shape
            y_coords = sorted_indices // w
            x_coords = sorted_indices % w
            
            # Process pixels in order of priority
            for idx in range(len(sorted_indices)):
                y, x = y_coords[idx].item(), x_coords[idx].item()
                
                # Try different adjustments for each channel
                best_score = float('-inf')
                best_adjustment = torch.zeros(3, device=self.device)
                
                for c in range(3):  # For each color channel
                    for adj in [-self.delta, 0, self.delta]:
                        # Create a temporary patch with this adjustment
                        temp_patch = patch.clone()
                        temp_patch[c, y, x] += adj
                        temp_patch.clamp_(0, 1)  # Keep in valid range
                        
                        # Evaluate this adjustment
                        score = self.evaluate_patch(image, temp_patch, true_label)
                        
                        if score > best_score:
                            best_score = score
                            best_adjustment[c] = adj
                
                # Apply the best adjustment to the patch
                for c in range(3):
                    patch[c, y, x] += best_adjustment[c]
                
                # Ensure patch values stay in valid range
                patch.clamp_(0, 1)
                
                # Log progress periodically
                if idx % 100 == 0:
                    self.logger.info(f"Processed {idx}/{len(sorted_indices)} pixels")
            
            # Evaluate the patch after processing this image
            with torch.no_grad():
                patched_image, patched_label = self.apply_patch(image, true_label, patch)
                output = self.model.predict(patched_image, patched_label.shape)
                self.metric.update(output, patched_label)
                pixAcc, mIoU = self.metric.get()
                
                self.logger.info(f"Image {i_iter+1} complete - mIoU: {mIoU:.4f}, pixAcc: {pixAcc:.4f}")
                IoU.append(self.metric.get(full=True))
                self.metric.reset()
        
        # Save the optimized patch if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(patch, save_path)
            self.logger.info(f"Saved optimized patch to {save_path}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Greedy optimization completed in {str(datetime.timedelta(seconds=int(total_time)))}")
        
        return patch.detach(), np.array(IoU)
