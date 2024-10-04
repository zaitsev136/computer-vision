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
    

def get_intersections_and_unions(predictions, targets, ignore_background=True):
    """Calculates intersections and unions of predicted and target segmentation
    maps for each class

    Args:
        predictions (torch.Tensor): prediction logits of shape (B, C, H, W)
        targets (torch.Tensor): segmentation map with class indices, (B, H, W)
        ignore_background (bool): If True, the first class is considered to be
            background and pixels marked as background in the targets are masked
            out of the calculation. Defaults to True.

    Returns:
        torch.Tensor, torch.Tensor: intersections and unions for each class.
            Shapes (C-1) if ignore_background is true, and (C) otherwise
    """
    num_classes = predictions.size()[1]
    predictions = torch.argmax(predictions, dim=1)  # from logits to class indices
    predictions = F.one_hot(predictions, num_classes=num_classes)  # from indices to one_hot (B, H, W, C)
    targets = F.one_hot(targets.to(torch.int64), num_classes=num_classes)  # from indices to one_hot (B, H, W, C)

    reduce_axis = [0, 1, 2]
    if ignore_background:
        intersection = torch.sum(predictions & targets, dim=reduce_axis)[1:]
        mask = ~targets[..., :1].repeat(1, 1, 1, num_classes).to(torch.bool)
        union = torch.sum(predictions*mask | targets*mask, dim=reduce_axis)[1:]
        return intersection, union
    else:
        intersection = torch.sum(predictions & targets, dim=reduce_axis)
        union = torch.sum(predictions | targets, dim=reduce_axis)
        return intersection, union
    

def get_accuracy(predictions, targets, ignore_background=True):
    """Calculates classification accuracy

    Args:
        predictions (torch.Tensor): prediction logits of shape (B, C, H, W)
        targets (torch.Tensor): segmentation map with class indices, (B, H, W)
        ignore_background (bool): If True, the first class is considered to be
            background and pixels marked as background in the targets are masked
            out of the calculation. Defaults to True.

    Returns:
        float: classification accuracy
    """
    predictions = torch.argmax(predictions, dim=1)  # from logits to class indices
    mask = targets!=0 if ignore_background else torch.ones_like(targets)
    return torch.sum(predictions==targets) / torch.sum(mask)


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
