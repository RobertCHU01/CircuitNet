import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from PIL import Image 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from swintransformer import *
import utils.losses as losses
import torch.optim as optim
from tqdm import tqdm
from math import cos, pi
from train import CosineRestartLr


# debugging functions
def print_grad_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")

def print_param_changes(model, step):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if step == 'before':
                param.data_before = param.data.clone()
            else:
                print(f"Change in {name}: {torch.norm(param.data - param.data_before)}")

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
            if all(os.path.exists(fp) for fp in feature_paths) and os.path.exists(label_path):
                self.data.append((feature_paths, label_path))
            i+=1
            if i>99:
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

def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/swinTransformer_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


feature_list = ['IR_drop_features_decompressed/power_i', 'IR_drop_features_decompressed/power_s', 
        'IR_drop_features_decompressed/power_sca', 'IR_drop_features_decompressed/power_all']
label_list = ['IR_drop_features_decompressed/IR_drop']

datapath = './CircuitNet-N28/'
# datapath = '../../CircuitNet/CircuitNet-N28/'
name_list = get_sub_path(os.path.join(datapath, feature_list[-1]))
n_list = divide_list(name_list, 1000)

    
root_dir = './CircuitNet-N28/IR_drop_features_decompressed/'
dataset = PowerDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# c=0
# for features, labels in dataloader:
#     print(features.shape, labels.shape)
#     if (c == 3): break
#     c+=1


model_name = 'swin_base_patch4_window7_224'
model = init_model(model_name, input_channels=4, num_classes=0, pretrained=True)
model.train()



# Build loss
# loss = losses.__dict__['L1Loss']()
loss = nn.L1Loss()

# arg_dict = {'task': 'irdrop_mavi', 'save_path': 'work_dir/irdrop_mavi/', 'pretrained': None, 'max_iters':200, 'plot_roc': False, 'arg_file': None, 'cpu': True, 'dataroot': 'CircuitNet-N28/training_set/IR_drop', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'IRDropDataset', 'batch_size': 2, 'model_type': 'MAVI', 'in_channels': 1, 'out_channels': 4, 'lr': 0.0002, 'weight_decay': 0.01, 'loss_type': 'L1Loss', 'eval_metric': ['NRMS', 'SSIM'], 'threshold': 0.9885, 'ann_file': './files/train_N28.csv', 'test_mode': False}
arg_dict = {'task': 'irdrop_mavi', 'save_path': 'work_dir/irdrop_mavi/', 'pretrained': None, 'max_iters':200, 'plot_roc': False, 'arg_file': None, 'cpu': True, 'dataroot': 'CircuitNet-N28/training_set/IR_drop', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'IRDropDataset', 'batch_size': 2, 'model_type': 'MAVI', 'in_channels': 1, 'out_channels': 4, 'lr': 0.01, 'weight_decay': 0.01, 'loss_type': 'L1Loss', 'eval_metric': ['NRMS', 'SSIM'], 'threshold': 0.9885, 'ann_file': './files/train_N28.csv', 'test_mode': False}
# # Build Optimzer
# optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Build lr scheduler
cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
cosine_lr.set_init_lr(optimizer)

epoch_loss = 0
iter_num = 0
print_freq = 100
# save_freq = 10000
save_freq = 1000
epoch_loss2 = 0


while iter_num < arg_dict['max_iters']:
    with tqdm(total=print_freq) as bar:
        # print(len(dataset))      497 batches
        for feature, label in dataloader:        
            if arg_dict['cpu']:
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            # regular_lr = cosine_lr.get_regular_lr(iter_num)
            # cosine_lr._set_lr(optimizer, regular_lr)

            prediction = model(input)
            # print(input.shape)

            optimizer.zero_grad()
            pixel_loss = loss(prediction, target)

            epoch_loss += pixel_loss.item()
            pixel_loss.backward()
            optimizer.step()

            iter_num += 1
            
            bar.update(1)   # bar increases by one when model goes through one batch in dataset instead of full dataset

            # print_freq = 100, which means for loop only goes through batches 0 - 99
            if iter_num % print_freq == 0:
                break
# My Code
# #------------------------------------------------------------------------------------
# # params all require grad
# for name, param in model.named_parameters():
#     if not param.requires_grad:
#         print(f"{name}: does not require_grad")

# with tqdm(total=print_freq) as bar:
#     while iter_num < arg_dict['max_iters']:
#         # for feature, label, _ in dataset: 
#         # count = 0      
#         epoch_loss2 = epoch_loss
# # Inside your training loop:
#         for feature, label in dataloader:
#             if arg_dict['cpu']:
#                 input, target = feature, label
#             else:
#                 input, target = feature.cuda(), label.cuda()

#             # Print learning rate
#             # print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

#             # # Print parameters before update
#             # print("Parameters before update:")
#             # print_param_changes(model, 'before')

#             prediction = model(input)
#             prediction = prediction.squeeze(1)
#             optimizer.zero_grad()
#             pixel_loss = loss(prediction, target)
#             epoch_loss += pixel_loss.item()

#             pixel_loss.backward()
#             # for name, param in model.named_parameters():
#             #     # grad = torch.autograd.grad(pixel_loss, param, retain_graph=True, allow_unused=True)[0]
#             #     if param.grad is not None:
#             #         print(f"Gradient for {name}: mean {param.grad.mean().item()}, std {param.grad.std().item()}")
#             #     else:
#             #         print(f"No gradient for {name}")
#             # Print gradient norms
#             # print_grad_norms(model)

#             optimizer.step()
#             # # Check gradients after optimizer step
#             # for name, param in model.named_parameters():
#             #     if param.grad is not None:
#             #         print(f"Gradient after step for {name}: mean {param.grad.mean().item()}, std {param.grad.std().item()}")
#             #     else:
#             #         print(f"No gradient after step for {name}")

#             # Print parameters after update
#             # print("Parameters after update:")
#             # print_param_changes(model, 'after')

#             # print(f"Loss: {pixel_loss.item()}")
#             # print("-" * 50)
#         print("Loss in one epoch: {}".format(epoch_loss - epoch_loss2))
#         iter_num += 1
            
#         bar.update(1)

#         if iter_num % print_freq == 0:
#             break
# #------------------------------------------------------------------------------------

    print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
    if iter_num % save_freq == 0:
        checkpoint(model, iter_num, arg_dict['save_path'])
    epoch_loss = 0
torch.save(model, 'full_model.pth')