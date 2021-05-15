import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import normalizeFeaturesL2

class SJE_GMPool(nn.Module):

    def __init__(self, img_feature_size, num_attributes, margin):
        super(SJE_GMPool, self).__init__()
        self.margin = margin

        # copying initialization technique from original code
        W = torch.rand(img_feature_size, num_attributes, requires_grad=True)
        W = normalizeFeaturesL2(W.permute(1,0)).permute(1,0)
        self.W = nn.Parameter(W, requires_grad=True)

        power = torch.zeros(num_attributes, requires_grad=True)
        self.power = nn.Parameter(power, requires_grad=True)
        self.example_indices = random.choices(range(1000), k=2) # this is a hack

    def get_power(self):
        c = float(10)
        p = self.power * 3
        power = torch.zeros_like(p)
        power[p>=2] = c
        power = torch.where((0<=p)&(p<2), (c-1)/2*p+1, power)
        power = torch.where((-1<=p)&(p<0), 1/((1-c)*p+1), power)
        power = torch.where((-1.5<=p)&(p<-1), -1/(2*(c-1)*(p+1.5)+1), power)
        power = torch.where((-2<=p)&(p<-1.5), 2*(c-1)*(p+2)-c, power)
        power[p<-2] = -c
        assert torch.all(power != 0)
        return power

    def apply_gmpool(self, projected_feats):
        '''
        projected_feats: torch.Tensor of shape [B, num_attributes, H, W]
        returns pooled features of shape [B, num_attributes]
        '''
        m = projected_feats.min()
        p = self.get_power().view(1,-1,1,1)
        if m < 0:
            pooled = (projected_feats-m+1e-3).pow(p).mean(2, keepdim=True).mean(3, keepdim=True).pow(1/p)+m+1e-3
        else:
            pooled = projected_feats.pow(p).mean(2, keepdim=True).mean(3, keepdim=True).pow(1/p)
        return pooled.squeeze(2).squeeze(2)

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
        XW = torch.tensordot(img_features, self.W, [[1],[0]]).permute(0,3,1,2) # shape [B, num_attributes, H, W]
        XW = self.apply_gmpool(XW) # shape [B, num_attributes]
        if torch.any(XW.isnan()):
            print("YIKES")
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        gt_class_scores = scores[torch.arange(len(scores)), labels].unsqueeze(1) # shape [B, 1]
        # add margin to scores
        losses = self.margin + scores - gt_class_scores # shape [B, num_classes]
        losses[torch.arange(len(losses)), labels] = 0.0
        losses = losses.max(dim=1)[0] # shape [B]
        return losses.clamp(0).mean()

    def forward_test(self, img_features, all_class_attributes):
        XW = torch.tensordot(img_features, self.W, [[1],[0]]).permute(0,3,1,2) # shape [B, num_attributes, H, W]
        XW = self.apply_gmpool(XW) # shape [B, num_attributes]
        if torch.any(XW.isnan()):
            print("YIKES")
        XW = normalizeFeaturesL2(XW) # normalize each projected vector to have unit length
        scores = torch.matmul(XW.unsqueeze(1), all_class_attributes).squeeze(1) # shape [B, num_classes]
        return scores.argmax(1) # shape [B]

    def log_spatial_examples(self, dataloader, device, writer, split, epoch):
        dataset = dataloader.dataset
        self.eval()
        classes = dataset.classes
        for i, idx in enumerate(self.example_indices):
            # unpack data
            data = dataset[idx]
            img_features = data['img'].to(device).unsqueeze(0)
            gt_label = classes[data['label']]
            all_class_attributes = dataset.class_attributes
            gt_class_attributes = all_class_attributes[:,data['label']]
            img = mpimg.imread(dataset.get_img_path(idx))

            # forward pass
            XW = torch.tensordot(img_features, self.W, [[1],[0]]).permute(0,3,1,2).squeeze() # shape [num_attributes, H, W]

            for spatial_dist, gt_attribute_score, attribute_name in zip(XW, gt_class_attributes, dataset.attributes):
                fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
                ax1.set_title(f"Attribute: {attribute_name}\nGT Attribute Value: {gt_attribute_score:.4f}")
                mappable = ax1.imshow(spatial_dist.cpu().detach().numpy(), vmin=-0.25, vmax=0.25)
                fig.colorbar(mappable, ax=ax1)
                ax2.set_title(f"Original Image({gt_label})")
                ax2.imshow(img)
                plt.tight_layout()
                writer.add_figure(f"Spatial Examples ({split})/{attribute_name}-{i}", fig, epoch)
                plt.close(fig)

