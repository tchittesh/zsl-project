import torch
import torch.nn as nn

from utils import normalizeFeaturesL2


class SJE_Original(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_Original, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        W = torch.rand(img_feature_size, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(1, 0)).permute(1, 0)
        self.W = nn.Parameter(W, requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
        img_features = self.avgpool(img_features).squeeze(2).squeeze(2)
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
        img_features = self.avgpool(img_features).squeeze(2).squeeze(2)
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        return scores.argmax(1)  # shape [B]


class SJE_Linear(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_Linear, self).__init__()
        self.margin = margin
        self.projection = nn.Linear(img_feature_size, num_attributes)
        self.projection.weight.data = normalizeFeaturesL2(self.projection.weight.data)

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
        XW = self.projection(img_features)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1)  # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores  # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0]  # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = self.projection(img_features)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        return scores.argmax(1)  # shape [B]


class SJE_MLP(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_MLP, self).__init__()
        self.margin = margin
        self.projection = nn.Sequential(
            nn.Linear(img_feature_size, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(256, num_attributes),
            nn.Dropout(p=0.8)
        )

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
        XW = self.projection(img_features)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1)  # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores  # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0]  # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = self.projection(img_features)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        return scores.argmax(1)  # shape [B]
