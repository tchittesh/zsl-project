import torch
import torch.nn as nn

from utils import normalizeFeaturesL2


class SJE_CosEmb(nn.Module):

    def __init__(self, class_attribute_emb, img_feature_size, num_attributes, margin):
        super(SJE_CosEmb, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        class_attr_emb = None
        W = torch.rand(img_feature_size, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(1, 0)).permute(1, 0)
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
        print(img_features.shape)
        print(class_attributes.shape)
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1)  # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores  # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0]  # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        return scores.argmax(1)  # shape [B]