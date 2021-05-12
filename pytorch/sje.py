import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import normalizeFeaturesL2

class SJE_Original(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_Original, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        W = torch.rand(img_feature_size, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(1,0)).permute(1,0)
        self.W = nn.Parameter(W, requires_grad=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        if len(img_features.shape) == 4:
            img_features = self.avg_pool(img_features).squeeze(2).squeeze(2) # remove h, w dimensions
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1) # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0] # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        if len(img_features.shape) == 4:
            img_features = self.avg_pool(img_features).squeeze(2).squeeze(2) # remove h, w dimensions
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        return scores.argmax(1) # shape [B]

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
        XW = self.projection(img_features) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1) # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0] # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = self.projection(img_features) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        return scores.argmax(1) # shape [B]

class SJE_WeightedCosine(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_WeightedCosine, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        W = torch.rand(img_feature_size, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(1,0)).permute(1,0)
        self.W = nn.Parameter(W, requires_grad=True)

        weights = torch.zeros(num_attributes, requires_grad=True)
        self.weights = nn.Parameter(weights, requires_grad=True)

    def get_weights(self):
        weights = self.weights + 1.0
        return weights / weights.sum() * len(weights) # normalize weights

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
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length

        XW = self.get_weights().unsqueeze(0) * XW

        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1) # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0] # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length

        XW = self.get_weights().unsqueeze(0) * XW

        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        return scores.argmax(1) # shape [B]

class SJE_MLP(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_MLP, self).__init__()
        self.margin = margin
        self.projection = nn.Sequential(
            nn.Linear(img_feature_size, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_attributes),
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
        XW = self.projection(img_features) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1) # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0] # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = self.projection(img_features) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        return scores.argmax(1) # shape [B]

class SJE_GMPool(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_GMPool, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        W = torch.rand(img_feature_size, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(1,0)).permute(1,0)
        self.W = nn.Parameter(W, requires_grad=True)

        power = torch.rand(num_attributes, requires_grad=True) + 0.5
        self.power = nn.Parameter(power, requires_grad=True)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, img_features, all_class_attributes, class_attributes, labels):
        '''
        img_features: torch.Tensor of shape [B, img_feature_size, H, W]
        class_attributes: torch.Tensor of shape [B, num_attributes]
        labels: torch.Tensor of shape [B]
        all_class_attributes: torch.Tensor of shape [num_attributes, num_classes]
        returns scalar loss
        '''
        img_features = img_features.permute(0,2,3,1).view()
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1) # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0] # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = torch.matmul(img_features.unsqueeze(1), self.W).squeeze(1) # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        return scores.argmax(1) # shape [B]

