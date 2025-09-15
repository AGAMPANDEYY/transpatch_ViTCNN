import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#### Segmentation Metric 1

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, config):
        super(SegmentationMetric, self).__init__()
        self.config = config
        self.nclass = config.dataset.num_classes
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        ## reshaping
        h, w = labels.size(1), labels.size(2)
        ph, pw = preds.size(2), preds.size(3)
        if ph != h or pw != w:
            preds = F.interpolate(preds, size=(
                h, w), mode='bilinear', align_corners=self.config.model.align_corners)


        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            #print(f'\ncorrect_pixels:{correct}')
            #print(f'\total_pixels:{labeled}')

            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self,full = False):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        #print(f'Iou:\n{IoU}')
        mIoU = IoU.mean().item()
        if full == True:
          return IoU
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    #print(output)
    #print(f'---------------This is Segmetric 1: Pix accuracy---------------')
    #print(f'Target\n: {target}')
    #print(f'Output\n: {output}')
    predict = torch.argmax(output, 1) + 1
    #print(f'Predict from output+1: {predict}')
    #print(predict)
    target = target.float() + 1 ## so as to avoid -1
    target[target>=255] = 0
    #print(f'Target after being updated\n:{target}')
    pixel_labeled = torch.sum(target > 0).item()
    #print(f'Pix labeled: {pixel_labeled}')
    try:
        pixel_correct = torch.sum((predict == target) * (target > 0)).item()
        #print(f'Pix Correct: {pixel_correct}')
    except:
        print("predict size: {}, target size: {}, ".format(predict.size(), target.size()))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass


    #print(f'---------------This is Segmetric 1: MIOU---------------')
    #print(f'Target\n: {target}')
    #print(f'Output\n: {output}')
    predict = torch.argmax(output, 1) + 1  # [N,H,W] values from 1 to 19
    #print(f'Predict from output+1: {predict}')
    #print(predict)
    target = target.float() + 1            # [N,H,W] values from 1 to 19 and 256.0
    target[target>=255] = 0   # values from 0 to 19, where 0 should be ignored
    #print(f'Target after being updated\n:{target}')

    predict = predict.float() * (target>0).float() # predict will be 0 to 19 now, with 0 where target = 0
    #print(f'Predict after putting 0 where target is 0\n:{predict}')
    intersection = predict * (predict == target).float() ## all correct prediction pixels and 0 pixels
    #print(f'Intersection\n:{intersection}')
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    #print(f'Area intersection\n:{area_inter}')
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    #print(f'Area Pred\n:{area_pred}')
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    #print(f'Area Target\n:{area_lab}')
    area_union = area_pred + area_lab - area_inter
    #print(f'Area Union\n:{area_union}')
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()



def pixAcc_miou(config,model,dataloader, device, name):
    model.eval()
    metric = SegmentationMetric(config)
    metric.reset()
    lsit_pixAcc = []
    list_mIoU = []
    mean_standard = np.array([0.485, 0.456, 0.406],dtype = np.float32)
    std_standard = np.array([0.229, 0.224, 0.225],dtype = np.float32)
    with torch.no_grad():
        for i_iter, batch in tqdm(enumerate(dataloader, 0)):
            image_standard, targets, bd_gts, _, _ = batch
            targets = targets.to(device)
            if 'pidnet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              size = targets.shape
              output = F.interpolate(
                            outputs[config.test.output_index_pidnet], size[-2:],
                            mode='bilinear', align_corners=True
                        )

            if 'segformer' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              size = targets.shape
              output = F.interpolate(
                            outputs.logits, size[-2:],
                            mode='bilinear', align_corners=True
                        )

            if 'icnet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              output = outputs[config.test.output_index_icnet]

            if 'bisenet' in name:
              ## Images needs to be unnormalized and then normalized as:
              ## mean=[0.3257, 0.3690, 0.3223], std=[0.2112, 0.2148, 0.2115]
              ## The it will give 75% miou instead of 71 and to keep things simple keeping it as it
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              output = outputs[config.test.output_index_bisenet]

            metric.update(output, targets)
    print(f'\nPixel Accuracy: {metric.get()[0]}')
    print(f'miou: {metric.get()[1]}')



#### Segmentation Metric 2

class SegmentationMetrics2:
    def __init__(self, config):
        """
        Initializes the segmentation metrics for computing mIoU and MPA.

        Args:
            num_classes (int): Number of classes.
            ignore_index (int, optional): Class index to ignore during computation.
        """
        self.num_classes = config.dataset.num_classes
        self.ignore_index = None
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.correct_pixels = torch.zeros(self.num_classes)
        self.total_pixels = torch.zeros(self.num_classes)

    def update(self, pred, target):
        """
        Updates the IoU and pixel accuracy metrics given a batch of predictions and targets.

        Args:
            pred (torch.Tensor): Predicted logits or probabilities with shape (N, C, H, W).
            target (torch.Tensor): Ground truth mask with shape (N, H, W).
        """
        pred_labels = torch.argmax(pred, dim=1)  # Convert logits to class indices (N, H, W)
        #pred_labels = pred_labels.float() * (target>0).float()
        pred_labels[target == 255] = 255.0  # where target is 255, predlabel is 255
        # pred_labels: 0 to 18 values
        # target: 0 to 18 and 255 values
        #print(f'print target and label of 2')
        #print(target)
        #print(pred_labels)
        for cls in range(self.num_classes):
            if self.ignore_index is not None and cls == self.ignore_index:
                continue

            pred_mask = (pred_labels == cls)
            target_mask = (target == cls)

            # IoU computation
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()

            self.intersection[cls] += intersection
            self.union[cls] += union

            # Pixel accuracy computation
            correct = (pred_labels[target == cls] == cls).sum().item()
            total = (target == cls).sum().item()

            self.correct_pixels[cls] += correct
            self.total_pixels[cls] += total

    def get(self):
        """
        Computes the final Mean IoU (mIoU) and Mean Pixel Accuracy (MPA) across all processed batches.

        Returns:
            tuple: (Mean IoU, Mean Pixel Accuracy)
        """


        total_correct = self.correct_pixels.sum().item()
        total_pixels = self.total_pixels.sum().item()
        mean_pixel_acc = total_correct / (total_pixels + 1e-10)

        ious = self.intersection / (self.union + 1e-10)  # Avoid division by zero
        mean_iou = torch.nanmean(ious).item()  # Compute mean IoU ignoring NaN values

        return mean_pixel_acc,mean_iou

    def reset(self):
        """Resets the metric for a new evaluation."""
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.correct_pixels = torch.zeros(self.num_classes)
        self.total_pixels = torch.zeros(self.num_classes)




def pixAcc_miou2(config,model,dataloader, device, name):
    model.eval()
    metric = SegmentationMetrics2(config)
    metric.reset()
    lsit_pixAcc = []
    list_mIoU = []
    mean_standard = np.array([0.485, 0.456, 0.406],dtype = np.float32)
    std_standard = np.array([0.229, 0.224, 0.225],dtype = np.float32)
    with torch.no_grad():
        for i_iter, batch in tqdm(enumerate(dataloader, 0)):
            image_standard, targets, bd_gts, _, _ = batch
            targets = targets.to(device)
            if 'pidnet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              size = targets.shape
              output = F.interpolate(
                            outputs[config.test.output_index_pidnet], size[-2:],
                            mode='bilinear', align_corners=True
                        )


            if 'icnet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              output = outputs[config.test.output_index_icnet]

            if 'bisenet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              output = outputs[config.test.output_index_bisenet]

            metric.update(output, targets)
            metric.get()
    print(f'\nPixel Accuracy: {metric.get()[0]}')
    print(f'miou: {metric.get()[1]}')


#### Segmentation Metric 3

class SegmentationMetrics3:
    def __init__(self, config):
        """
        Initializes the segmentation metrics for computing mIoU and MPA.

        Args:
            num_classes (int): Number of classes.
            ignore_index (int, optional): Class index to ignore during computation.
        """
        self.num_classes = config.dataset.num_classes
        self.ignore_index = 0
        self.intersection = torch.zeros(self.num_classes+1)
        self.union = torch.zeros(self.num_classes+1)
        self.correct_pixels = torch.zeros(self.num_classes+1)
        self.total_pixels = torch.zeros(self.num_classes+1)

    def update(self, pred, target):
        """
        Updates the IoU and pixel accuracy metrics given a batch of predictions and targets.

        Args:
            pred (torch.Tensor): Predicted logits or probabilities with shape (N, C, H, W).
            target (torch.Tensor): Ground truth mask with shape (N, H, W).
        """

        #print(f'---------------This is Segmetric 3: ---------------')
        pred_labels = torch.argmax(pred, dim=1) + 1  # Convert logits to class indices (N, H, W)
        #print(f'Final predict + 1:\n:{pred_labels}')
        target = target.float() + 1
        # pred_labels: 1 to 19 values
        # target: 1 to 19 and 256 values
        target[target>=255] = 0
        pred_labels = pred_labels.float() * (target>0).float() ## where target is 0, predlabel is 0
        #print(f'Target +1 after changing 255 to 0\n:{target}')
        # target: 0 to 19 and 0 should be ignore index
        #print(f'print target and label of 3')
        #print(target)
        #print(pred_labels)
        for cls in range(self.num_classes+1):
            if self.ignore_index is not None and cls == self.ignore_index:
                continue

            #print(f'Class:{cls}')

            pred_mask = (pred_labels == cls)
            target_mask = (target == cls)

            # IoU computation
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()

            self.intersection[cls] += intersection
            self.union[cls] += union

            # Pixel accuracy computation
            correct = (pred_labels[target == cls] == cls).sum().item()
            total = (target == cls).sum().item()

            self.correct_pixels[cls] += correct
            self.total_pixels[cls] += total

        #print(f'Intersection:\n:{self.intersection}')
        #print(f'Union:\n:{self.union}')
        #print(f'correct_pixels:\n{self.correct_pixels}')
        #print(f'total_pixels:\n{self.total_pixels}')

    def get(self):
        """
        Computes the final Mean IoU (mIoU) and Mean Pixel Accuracy (MPA) across all processed batches.

        Returns:
            tuple: (Mean IoU, Mean Pixel Accuracy)
        """
        total_correct = self.correct_pixels[1:].sum().item()
        total_pixels = self.total_pixels[1:].sum().item()
        mean_pixel_acc = total_correct / (total_pixels + 1e-10)


        ious = self.intersection / (self.union + 1e-10)  # Avoid division by zero
        mean_iou = torch.nanmean(ious[1:]).item()  # Compute mean IoU ignoring NaN values


        return mean_pixel_acc,mean_iou

    def reset(self):
        """Resets the metric for a new evaluation."""
        self.intersection = torch.zeros(self.num_classes+1)
        self.union = torch.zeros(self.num_classes+1)
        self.correct_pixels = torch.zeros(self.num_classes+1)
        self.total_pixels = torch.zeros(self.num_classes+1)




def pixAcc_miou3(config,model,dataloader, device, name):
    model.eval()
    metric = SegmentationMetrics3(config)
    metric.reset()
    lsit_pixAcc = []
    list_mIoU = []
    mean_standard = np.array([0.485, 0.456, 0.406],dtype = np.float32)
    std_standard = np.array([0.229, 0.224, 0.225],dtype = np.float32)
    with torch.no_grad():
        for i_iter, batch in tqdm(enumerate(dataloader, 0)):
            image_standard, targets, bd_gts, _, _ = batch
            targets = targets.to(device)
            if 'pidnet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              size = targets.shape
              output = F.interpolate(
                            outputs[config.test.output_index_pidnet], size[-2:],
                            mode='bilinear', align_corners=True
                        )


            if 'icnet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              output = outputs[config.test.output_index_icnet]

            if 'bisenet' in name:
              image_standard = image_standard.to(device)
              outputs = model(image_standard)
              output = outputs[config.test.output_index_bisenet]

            metric.update(output, targets)
    print(f'\nPixel Accuracy: {metric.get()[0]}')
    print(f'miou: {metric.get()[1]}')

