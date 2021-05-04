import torch
import torch.nn as nn

from utils import normalizeFeaturesL2


class SJE_Spatial(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_Spatial, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        W = torch.rand(img_feature_size, 7, 7, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        self.W = nn.Parameter(W, requires_grad=True)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, img_features, all_class_attributes, class_attributes, labels):
        '''
        img_features: torch.Tensor of shape [B, img_feature_size]
        class_attributes: torch.Tensor of shape [B, num_attributes]
        labels: torch.Tensor of shape [B]
        all_class_attributes: torch.Tensor of shape [num_attributes, num_classes]
        returns scalar loss
        '''
        XW = torch.tensordot(img_features, self.W, dims=((1,2,3),(0,1,2))) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1)  # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores  # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0]  # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = torch.tensordot(img_features, self.W, dims=((1, 2, 3), (0, 1, 2)))  # shape [B, num_attributes
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        return scores.argmax(1)  # shape [B]
