import os
import numpy as np
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from swintransformer import init_model
from utils.losses import build_loss
from utils.configs import Parser
from math import cos, pi
from train import CosineRestartLr as BaseCosineRestartLr
import torch.nn.functional as F

class PowerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, target_size=(224, 224), max_samples=400):
        self.root_dir = root_dir
        self.feature_dirs = ['power_i', 'power_s', 'power_sca', 'Power_all']
        self.label_dir = 'IR_drop'
        self.target_size = target_size
        self.data = []
        for i, case_name in enumerate(os.listdir(os.path.join(root_dir, self.feature_dirs[0]))):
            feature_paths = [os.path.join(root_dir, feature_dir, case_name) for feature_dir in self.feature_dirs]
            label_path = os.path.join(root_dir, self.label_dir, case_name)
            if all(os.path.exists(fp) for fp in feature_paths) and os.path.exists(label_path):
                self.data.append((feature_paths, label_path))
            if i >= max_samples - 1:
                break

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_paths, label_path = self.data[idx]
        features = []
        
        for fp in feature_paths:
            feature = np.load(fp)
            feature = torch.tensor(feature, dtype=torch.float32)
            feature = F.interpolate(feature.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='nearest').squeeze()
            feature = self.normalize(feature)
            features.append(feature)
            
        features = torch.stack(features, dim=0)
        
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.float32)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='nearest').squeeze()
        label = label.clamp(1e-6, 50)
        label = (torch.log10(label) + 6) / (np.log10(50) + 6)
        
        return features, label
    
    @staticmethod
    def normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)

def build_dataset(arg_dict):
    return PowerDataset(arg_dict['dataroot'])

class CosineRestartLr(BaseCosineRestartLr):
    def __init__(self, base_lr, periods, restart_weights=[1], min_lr=None):
        super().__init__(base_lr, periods, restart_weights, min_lr)
        self.cumulative_periods = [sum(self.periods[:i+1]) for i in range(len(self.periods))]

    def get_lr(self, iter_num, base_lr):
        if iter_num >= self.cumulative_periods[-1]:
            return self.min_lr or 0.0
        return super().get_lr(iter_num, base_lr)

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, base_lr) for base_lr in self.base_lr]

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [group['initial_lr'] for group in optimizer.param_groups]


def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")

def train():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    arg_dict = {'task': 'irdrop_mavi', 'save_path': 'work_dir/irdrop_mavi/', 'pretrained': None, 'max_iters': 2000, 'plot_roc': False, 'arg_file': None, 'cpu': True, 'dataroot': 'CircuitNet-N28/training_set/IR_drop', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'IRDropDataset', 'batch_size': 2, 'model_type': 'MAVI', 'in_channels': 1, 'out_channels': 4, 'lr': 0.0002, 'weight_decay': 0.01, 'loss_type': 'L1Loss', 'eval_metric': ['NRMS', 'SSIM'], 'threshold': 0.9885, 'ann_file': './files/train_N28.csv', 'test_mode': False}
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'], 'arg.json'), 'wt') as f:
        json.dump(arg_dict, f, indent=4)

    print('===> Loading datasets')
    dataset = build_dataset(arg_dict)
    dataloader = DataLoader(dataset, batch_size=arg_dict['batch_size'], shuffle=True)

    print('===> Building model')
    model = init_model('swin_base_patch4_window7_224', input_channels=4, num_classes=0, pretrained=True)
    if not arg_dict['cpu']:
        model = model.cuda()
    
    loss_fn = build_loss(arg_dict)
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'], betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 1000

    model.train()
    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label in dataloader:
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                for param_group, lr in zip(optimizer.param_groups, regular_lr):
                    param_group['lr'] = lr

                prediction = model(input)
                optimizer.zero_grad()
                pixel_loss = loss_fn(prediction, target)
                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                iter_num += 1
                bar.update(1)

                if iter_num % print_freq == 0:
                    break

        print(f"===> Iters[{iter_num}]({iter_num}/{arg_dict['max_iters']}): Loss: {epoch_loss / print_freq:.4f}")
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path'])
        epoch_loss = 0

if __name__ == "__main__":
    train()