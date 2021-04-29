import random
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm

from dataset import ZSLDataset
from sje import SJE_Original

parser = argparse.ArgumentParser()

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-mr', '--margin', default=1, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)


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
    return num_correct / num_total


def main(args):
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    train_dataset = ZSLDataset(args.dataset, 'train', norm_type=args.norm_type)
    norm_info = train_dataset.norm_info if hasattr(train_dataset, 'norm_info') else None
    val_dataset = ZSLDataset(args.dataset, 'val', norm_type=args.norm_type, norm_info=norm_info)
    test_dataset = ZSLDataset(args.dataset, 'test', norm_type=args.norm_type, norm_info=norm_info)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SJE_Original(train_dataset.get_img_feature_size(), train_dataset.get_num_attributes(), margin=args.margin).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_val_ep = -1
    best_train_ep = -1
    best_params = None
    
    for ep in tqdm(range(args.epochs)):
        start = time.time()

        train(model, train_dataloader, optimizer, device)
        print(time.time() - start)
        train_acc = evaluate(model, train_dataloader, device)
        print(time.time() - start)
        val_acc = evaluate(model, val_dataloader, device)
        print(time.time() - start)

        end = time.time()
        elapsed = end-start

        print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, train_acc, val_acc, elapsed//60, elapsed%60))

        if val_acc>best_val_acc:
            best_val_acc = val_acc
            best_val_ep = ep+1
            best_params = model.state_dict()
        
        if train_acc>best_train_acc:
            best_train_ep = ep+1
            best_train_acc = train_acc

        if ep+1-best_val_ep>args.early_stop:
            print('Early Stopping by {} epochs. Exiting...'.format(args.epochs-(ep+1)))
            break

    print('\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_train_acc, best_train_ep))

    assert best_params is not None
    model.load_state_dict(best_params)
    test_acc = evaluate(model, test_dataloader, device)
    print('Test Acc:{}'.format(test_acc))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
