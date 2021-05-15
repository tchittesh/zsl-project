import random
import argparse
import time
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from dataset import ZSLDataset, ZSLSpatialDataset
from sje import SJE_Original, SJE_Linear, SJE_MLP, SJE_WeightedCosine
from sje_cos_emb import SJE_CosEmb
from sje_mha import SJE_MHA
from sje_gmpool import SJE_GMPool

model_dict = {
    'Original': SJE_Original,
    'MLP': SJE_MLP,
    'Linear': SJE_Linear,
    'WeightedCosine': SJE_WeightedCosine,
    'MHA': SJE_MHA,
    'GMPool': SJE_GMPool,
}

parser = argparse.ArgumentParser()

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-m', '--model', help=f'choose between {list(model_dict.keys())}', default='cos_emb', type=str)
parser.add_argument('-e', '--epochs', default=10, type=int) #CHANGE THIS
parser.add_argument('-s', '--spatial', action='store_true')
parser.add_argument('-l', '--logging', action='store_true')
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='L2', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)#CHANGE THIS
parser.add_argument('-mr', '--margin', default=1, type=float)#CHANGE THIS
parser.add_argument('-seed', '--rand_seed', default=42, type=int)#CHANGE THIS


def train(model, dataloader, optimizer, device):
    '''Train model on dataloader for one epoch. Returns avg loss.'''
    model.train()
    avg_loss = 0.0
    len_data = 0
    for batch_data in dataloader:
        # unpack data
        img_features = batch_data['img'].to(device)
        labels = batch_data['label'].to(device)
        class_attributes = batch_data['class_attributes'].to(device)
        all_class_attributes = dataloader.dataset.class_attributes.to(device)
        B = len(img_features)

        # forward and backward pass
        optimizer.zero_grad()
        loss = model(img_features, all_class_attributes, class_attributes, labels)
        loss.backward()
        optimizer.step()

        avg_loss += B * loss
        len_data += B
    return avg_loss / len_data


def evaluate(model, dataloader, device):
    '''Evaluate model on a dataset. Returns accuracy.'''
    model.eval()
    num_correct = 0
    num_total = 0
    len_data = 0
    y_true = []
    y_pred = []
    classes = dataloader.dataset.classes
    for batch_data in dataloader:
        # unpack data
        img_features = batch_data['img'].to(device)
        labels = batch_data['label'].to(device)
        all_class_attributes = dataloader.dataset.class_attributes.to(device)
        B = len(img_features)

        # forward pass
        preds = model(img_features, all_class_attributes)

        num_correct += (labels == preds).float().sum()
        num_total += B
        y_true.extend(classes[labels.cpu().detach().numpy()].tolist())
        y_pred.extend(classes[preds.cpu().detach().numpy()].tolist())
    accuracy = num_correct / num_total
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    per_cls_acc = ''
    for cls, total, num_correct in zip(classes, cm, cm.diagonal()):
        per_cls_acc += f'{cls:20}  {num_correct/sum(total)}  \n'
    return accuracy, cm_display, per_cls_acc

def log_confusion_matrix(cm_disp, writer, tag, epoch):
    fig, ax = plt.subplots(figsize=(8,8))
    cm_disp.plot(ax=ax, xticks_rotation='vertical', include_values=False)
    plt.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)

def log_example_images(model, dataloader, device, writer, split, epoch):
    dataset = dataloader.dataset
    indices = random.choices(range(len(dataset)), k=5)
    model.eval()
    classes = dataset.classes
    for i, idx in enumerate(indices):
        # unpack data
        data = dataset[idx]
        img_features = data['img'].to(device).unsqueeze(0)
        gt_label = classes[data['label']]
        all_class_attributes = dataset.class_attributes.to(device)
        img = mpimg.imread(dataset.get_img_path(idx))

        # forward pass
        pred_label = classes[model(img_features, all_class_attributes).squeeze(0)]
        
        fig, ax = plt.subplots()
        ax.set_title(f"Pred: {pred_label}, GT: {gt_label}")
        ax.imshow(img)
        plt.tight_layout()
        writer.add_figure(f"Example Images ({split})/{i}", fig, epoch)
        plt.close(fig)

def main(args):
    if args.rand_seed is not None:
        random.seed(args.rand_seed)
        np.random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)

    if args.spatial:
        train_dataset = ZSLSpatialDataset(args.dataset, 'train', norm_type=args.norm_type)
        norm_info = train_dataset.norm_info if hasattr(train_dataset, 'norm_info') else None
        val_dataset = ZSLSpatialDataset(args.dataset, 'val', norm_type=args.norm_type, norm_info=norm_info)
        test_dataset = ZSLSpatialDataset(args.dataset, 'test', norm_type=args.norm_type, norm_info=norm_info)
    else:
        train_dataset = ZSLDataset(args.dataset, 'train', norm_type=args.norm_type)
        norm_info = train_dataset.norm_info if hasattr(train_dataset, 'norm_info') else None
        val_dataset = ZSLDataset(args.dataset, 'val', norm_type=args.norm_type, norm_info=norm_info)
        test_dataset = ZSLDataset(args.dataset, 'test', norm_type=args.norm_type, norm_info=norm_info)

    print("Loaded datasets!")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_dict[args.model](train_dataset.get_img_feature_size(), train_dataset.get_num_attributes(), margin=args.margin).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_val_ep = -1
    best_train_ep = -1
    best_params = None
    if args.logging:
        writer = SummaryWriter()
        matplotlib.rc("font", size=8)

    for ep in tqdm(range(args.epochs)):
        start = time.time()

        tr_loss = train(model, train_dataloader, optimizer, device)
        if args.model == "GMPool":
            print(model.power)
            print(model.get_power())
        elif args.model == "WeightedCosine":
            print(model.weights)
            print(model.get_weights())

        print("Training done in ", time.time() - start, " Loss is :", tr_loss.item())
        train_acc, train_cm_disp, train_cls_acc = evaluate(model, train_dataloader, device)
        val_acc, val_cm_disp, val_cls_acc = evaluate(model, val_dataloader, device)

        if args.logging:
            writer.add_scalar('Loss/train', tr_loss, ep)
            writer.add_scalar('Accuracy/train', train_acc, ep)
            log_confusion_matrix(train_cm_disp, writer, 'Confusion Matrix/train', ep)
            writer.add_text('Class Accuracy/train', train_cls_acc, ep)
            log_example_images(model, train_dataloader, device, writer, 'train', ep)
            writer.add_scalar('Accuracy/val', val_acc, ep)
            log_confusion_matrix(val_cm_disp, writer, 'Confusion Matrix/val', ep)
            writer.add_text('Class Accuracy/val', val_cls_acc, ep)
            log_example_images(model, val_dataloader, device, writer, 'val', ep)

        end = time.time()
        elapsed = end - start

        print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep + 1, train_acc, val_acc,
                                                                                        elapsed // 60, elapsed % 60))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_ep = ep+1
            best_params = deepcopy(model.state_dict())

        if train_acc > best_train_acc:
            best_train_ep = ep + 1
            best_train_acc = train_acc

        if ep + 1 - best_val_ep > args.early_stop:
            print('Early Stopping by {} epochs. Exiting...'.format(args.epochs - (ep + 1)))
            break

    print(
        '\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_train_acc,
                                                                              best_train_ep))

    assert best_params is not None
    model.load_state_dict(best_params)
    test_acc, test_cm_disp, test_cls_acc = evaluate(model, test_dataloader, device)
    if args.logging:
        writer.add_scalar('Accuracy/test', test_acc, best_val_ep)
        log_confusion_matrix(test_cm_disp, writer, 'Confusion Matrix/test', ep)
        writer.add_text('Class Accuracy/test', test_cls_acc, ep)
        log_example_images(model, test_dataloader, device, writer, 'test', ep)
    print('Test Acc:{}'.format(test_acc))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
