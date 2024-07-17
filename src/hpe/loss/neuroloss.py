from torch.nn.modules.loss import _Loss
from torch import nn
import torch

metric_dict = {
    'MSE': nn.functional.mse_loss,
    'MAE': nn.functional.l1_loss,
    'SmoothL1': nn.functional.smooth_l1_loss,
    'CrossEntropy': nn.functional.cross_entropy,
    'CosineSim': lambda input, target, reduction=None: 1.-torch.cos(torch.deg2rad(input-target))
}

class NeuroLoss(_Loss):
    def __init__(self, metric='MSE', keypoints=None, weights=None):
        super(NeuroLoss, self).__init__()
        self.metric = metric_dict[metric]
        self.keypoints = keypoints
        self.weights = weights  # Add weights for different variables
        
    def forward(self, input, target):
        
        if len(input.shape) == 2:
            B, C = input.shape
            target = target[:,-1,:] # only take the last time step
            target = target.view(B,-1)
        
        assert input.shape == target.shape, "Input and target must have the same shape"

        loss = self.metric(input, target, reduction='none')
        
        # Apply different weights for different variables if weights are provided
        if self.weights is not None:
            weighted_loss = loss * self.weights.view(1, -1)  # Assuming weights is a tensor
            loss = weighted_loss.mean(dim=0)
        else:
            loss = loss.mean(dim=0)
        
        if len(target.shape) == 3:
            smoothness_loss = nn.functional.smooth_l1_loss(input[:, :-1, :], input[:, 1:, :])
            return loss.mean(dim=0), loss.mean() + smoothness_loss
        else:
            return loss, loss.mean()

if __name__ == "__main__":

    loss = NeuroLoss()
    input = torch.randn(3, 16)
    target = torch.randn(3, 16)

    test = loss(input, target)
    print(test)