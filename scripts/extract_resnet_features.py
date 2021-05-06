import torchvision
from torchvision.models import resnet101
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import os

save_dir = "."
transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
image_encoder = resnet101(pretrained=True)
image_encoder = torch.nn.Sequential(*(list(image_encoder.children())[:-2]))
dataset = torchvision.datasets.ImageFolder(root='Animals_with_Attributes2/JPEGImages',
                                           transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                               num_workers=0)
for c in dataset.classes:
    final_dir = f"{save_dir}/resnet_feats/{c}"
    if not os.path.exists(final_dir):
        os.makedirs(final_dir, ex)


for idx, (img, label) in tqdm(enumerate(train_dataloader)):
    feat = image_encoder(img)
    img_name = dataset.imgs[idx][0].split("/")[-1]
    ft_name = img_name.replace(".jpg", ".pt")
    torch.save(ft_name, f"{save_dir}/resent_feats/{dataset.classes[label.item()]}/{ft_name}")
print()
