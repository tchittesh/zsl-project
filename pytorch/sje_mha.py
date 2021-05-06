import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalizeFeaturesL2


class SJE_MHA(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_MHA, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        W = torch.rand(img_feature_size, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(1, 0)).permute(1, 0)
        self.W = nn.Parameter(W, requires_grad=True)
        self.attn = torch.nn.MultiheadAttention(img_feature_size, num_heads=1, dropout=0.5, kdim=num_attributes,
                                                vdim=num_attributes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        spW = torch.rand(img_feature_size, 7, 7, num_attributes, requires_grad=True)
        spW = normalizeFeaturesL2(spW.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        self.spW = nn.Parameter(spW, requires_grad=True)

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

        avg_feats = self.avgpool(img_features).squeeze()
        bs, channels = img_features.shape[0], img_features.shape[1]
        img_features = img_features.permute(2, 3, 0, 1)
        img_features = img_features.view(-1, bs, channels)
        XW = torch.matmul(avg_feats.unsqueeze(1), self.W).squeeze(1)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        XW = XW.unsqueeze(0)  # shape [1, B, num_attributes] This is key
        class_attributes = class_attributes.unsqueeze(0)  # shape [1, B, num_attributes] This is value
        att, att_weights = self.attn(query=img_features, key=XW, value=class_attributes)
        img_features = att + img_features
        img_features = img_features.view(self.spW.shape[1], self.spW.shape[1], bs, channels)
        img_features = img_features.permute(2, 3, 0, 1)
        finalW = torch.tensordot(img_features, self.spW, dims=((1, 2, 3), (0, 1, 2)))  # shape [B, num_attributes]
        finalW = normalizeFeaturesL2(finalW)

        scores = torch.matmul(finalW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1)  # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores  # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0]  # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        avg_feats = self.avgpool(img_features).squeeze()
        bs, channels = img_features.shape[0], img_features.shape[1]
        img_features = img_features.permute(2, 3, 0, 1)
        img_features = img_features.view(-1, bs, channels)
        XW = torch.matmul(avg_feats.unsqueeze(1), self.W).squeeze(1)  # shape [B, num_attributes]
        XW = normalizeFeaturesL2(XW)  # normalize each projected vector to have unit length
        XW = XW.unsqueeze(0)
        XW = torch.tile(XW, (all_class_attributes.shape[-1], 1, 1))  # shape [1, B, num_attributes] This is key
        att_class_attributes = torch.tile(all_class_attributes.permute(1, 0).unsqueeze(1),
                                          (1, bs, 1))  # shape [1, B, num_attributes] This is value
        att, att_weights = self.attn(query=img_features, key=XW, value=att_class_attributes)
        img_features = att + img_features
        img_features = img_features.view(self.spW.shape[1], self.spW.shape[1], bs, channels)
        img_features = img_features.permute(2, 3, 0, 1)
        finalW = torch.tensordot(img_features, self.spW, dims=((1, 2, 3), (0, 1, 2)))  # shape [B, num_attributes]
        finalW = normalizeFeaturesL2(finalW)

        scores = torch.matmul(finalW.unsqueeze(1), all_class_attributes).squeeze(1)  # shape [B, num_classes]
        return scores.argmax(1)  # shape [B]
