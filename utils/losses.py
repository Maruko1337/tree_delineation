import torch.nn.functional as F

def edge_loss(pred, target):
    # Custom implementation for edge loss
    return F.mse_loss(pred, target)

def region_loss(pred, target):
    return F.mse_loss(pred, target)

def occlusion_aware_loss(pred, target):
    # Custom implementation for occlusion-aware loss
    return F.mse_loss(pred, target)
