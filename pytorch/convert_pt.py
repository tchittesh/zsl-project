import torchvision
from torchvision.models import resnet101
import torchvision.transforms as transforms
import torch
from tqdm.notebook import tqdm
import os

save_dir = ""
transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
image_encoder = resnet101(pretrained=True)
print(image_encoder.training)
image_encoder.eval()
print(image_encoder.training)
image_encoder = torch.nn.Sequential(*(list(image_encoder.children())[:-2]))
dataset = torchvision.datasets.ImageFolder(root='/content/JPEGImages',
                                           transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                               num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_encoder.to(device)
# for c in dataset.classes:
# /content/drive/MyDrive/vlr_resnet/resnet_feats
#     final_dir = f"{save_dir}/resnet_feats/{c}"
#     os.makedirs(final_dir, exist_ok=True)

for idx, (img, label) in tqdm(enumerate(train_dataloader)):
    img = img.to(device)
    feat = image_encoder(img).squeeze()
    img_name = dataset.imgs[idx][0].split("/")[-1]
    ft_name = img_name.replace(".jpg", ".pt")
    torch.save(feat, f"{save_dir}/resnet_feats/{dataset.classes[label.item()]}/{ft_name}")
