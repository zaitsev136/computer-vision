import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torchmetrics.segmentation


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        """Dice loss

        Args:
            epsilon (float, optional): epsilon. Defaults to 1e-6.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """Calculates the dice loss.

        Args:
            predictions (torch.Tensor): prediction logits of shape (B, C, H, W)
            targets (torch.Tensor): segmentation map with class indices, (B, H, W)

        Returns:
            float: dice loss
        """
        
        num_classes = predictions.size(1)
        predictions = F.softmax(predictions, dim=1)  # Apply softmax to get class probabilities
        targets = rearrange(F.one_hot(targets.to(torch.int64), num_classes),
                            'b h w c -> b c h w')  # convert to one-hot 

        intersection = (predictions * targets).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)
        
        return 1 - dice_score.mean()  # Return Dice Loss
    
    
class MeanIoU(nn.Module):
    def __init__(self, num_classes, include_background=True, per_class=False):
        """Mean Intersection over Union metric

        Args:
            num_classes (int): number of segmentation classes
            include_background (bool, optional): whether or not to include
                background. Defaults to True.
            per_class (bool, optional): whether to compute the metric
                for each class separately or for all classes at once.
                Defaults to False.
        """
        super().__init__()
        self.metric = torchmetrics.segmentation.MeanIoU(num_classes,
                                                        include_background,
                                                        per_class)
    
    def forward(self, predictions, targets):
        """Calculates the MeanIoU metric.

        Args:
            predictions (torch.Tensor): prediction logits of shape (B, C, H, W)
            targets (torch.Tensor): segmentation map with class indices, (B, H, W)

        Returns:
            float or np.ndarray: meanIoU for each class separately or for
                all classes at once 
        """
        # takes prediction logits
        return self.metric(torch.argmax(predictions, dim=1), targets.to(torch.int64))


def albumentation_transform(transforms, x, y):
    """Applies the albumentations transforms to images and segmentation maps

    Args:
        transforms: albumentations transformation
        x (np.ndarray): images
        y (np.ndarray): segmentation masks

    Returns:
        tuple: transformed_images, transformed_masks
    """
    transformed = transforms(image=x, mask=y)
    return  transformed['image'], transformed['mask']
