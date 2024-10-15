# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser
from swintransformer import *
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

class PowerDataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224)):
        self.root_dir = root_dir
        self.feature_dirs = ['power_i', 'power_s', 'power_sca', 'Power_all']
        self.label_dir = 'IR_drop'
        self.target_size = target_size
        # Collect all the feature and label file paths
        self.data = []
        i=0
        for case_name in os.listdir(os.path.join(root_dir, self.feature_dirs[0])):
            feature_paths = [os.path.join(root_dir, feature_dir, case_name) for feature_dir in self.feature_dirs]
            label_path = os.path.join(root_dir, self.label_dir, case_name)
            i+=1
            # select 100 - 199
            if i>99:
                if all(os.path.exists(fp) for fp in feature_paths) and os.path.exists(label_path):
                    self.data.append((feature_paths, label_path))
            elif i > 199:
                break

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_paths, label_path = self.data[idx]
        features = []
        
        for fp in feature_paths:
            feature = np.load(fp)
            feature = torch.tensor(feature, dtype=torch.float32)
            feature = F.interpolate(feature.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0).squeeze(0)
            feature = std(feature)
            features.append(feature)
            
        features = torch.stack(features, dim=0)
        
        # Load and process label file
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.float32)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0).squeeze(0)
        label = label.clamp(1e-6, 50)
        label = (torch.log10(label)+6)/ (np.log10(50)+6)
        
        return features, label

def get_sub_path(path):
    sub_path = []
    if isinstance(path, list):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p, file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path, file))
    return sub_path

def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]
        
def std(input):
    if input.max() == 0:
        return input
    else:
        result = (input-input.min()) / (input.max()-input.min())
        return result

def test():
    total_loss = 0
    correct = 0
    total = 0

    # arg_dict = {'task': 'irdrop_mavi', 'save_path': 'work_dir/irdrop_mavi/', 'pretrained': None, 'max_iters':200, 'plot_roc': False, 'arg_file': None, 'cpu': True, 'dataroot': 'CircuitNet-N28/training_set/IR_drop', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'IRDropDataset', 'batch_size': 2, 'model_type': 'MAVI', 'in_channels': 1, 'out_channels': 4, 'lr': 0.0002, 'weight_decay': 0.01, 'loss_type': 'L1Loss', 'eval_metric': ['NRMS', 'SSIM'], 'threshold': 0.9885, 'ann_file': './files/train_N28.csv', 'test_mode': False}
    arg_dict = {'task': 'irdrop_mavi', 'save_path': 'work_dir/irdrop_mavi/', 'pretrained': None, 'max_iters':200, 'plot_roc': False, 'arg_file': None, 'cpu': True, 'dataroot': 'CircuitNet-N28/training_set/IR_drop', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'IRDropDataset', 'batch_size': 2, 'model_type': 'MAVI', 'in_channels': 1, 'out_channels': 4, 'lr': 0.0002, 'weight_decay': 0.01, 'loss_type': 'L1Loss', 'eval_metric': ['NRMS', 'SSIM'], 'threshold': 0.9885, 'ann_file': './files/train_N28.csv', 'test_mode': False}
    arg_dict['ann_file'] = arg_dict['ann_file_test'] 
    arg_dict['test_mode'] = True
    # print(arg_dict)
    print('===> Loading datasets')
    # Initialize dataset
    root_dir = './CircuitNet-N28/IR_drop_features_decompressed/'
    dataset = PowerDataset(root_dir)
    # Create a test dataset with the last 100 datapoints
    test_dataset = Subset(dataset, range(len(dataset) - 100, len(dataset)))
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    print('===> Building model')
    # Initialize model parameters
    # model = build_model(arg_dict)

    model_name = 'swin_base_patch4_window7_224'
    model = init_model(model_name, input_channels=4, num_classes=0, pretrained=True)
    model.load_state_dict(torch.load('full_model.pth'))
    # model = torch.load('full_model.pth')
    if not arg_dict['cpu']:
        model = model.cuda()

    model.eval()

    with tqdm(total=len(dataloader)) as bar:
        torch.no_grad()
        print(f"Number of batches in dataloader: {len(dataloader)}")
        print(f"Number of batches in dataset: {len(dataset)}")
        for feature, label in dataloader:
            try:
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                prediction = model(input)

                criterion = nn.L1Loss() 
                prediction = prediction.squeeze(1)
                loss = criterion(prediction, target)
                total_loss += loss.item()*100
                bar.update(1)
            except Exception as e:
                print(f"Error occurred at iteration {bar.n}: {str(e)}")
                break


    avg_loss = total_loss / len(dataloader)
    print("===> Avg. {:.4f}".format(avg_loss))




if __name__ == "__main__":
    test()