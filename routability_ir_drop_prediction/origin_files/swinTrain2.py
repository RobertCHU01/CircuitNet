import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from swintransformer import *
from train import CosineRestartLr
import math
from utils.losses import build_loss

class SimpleModel(nn.Module):
    def __init__(self, input_channels=4):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.leaky_relu()
        x = self.conv3(x)
        return x.squeeze(1)
    
class PowerDataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224), max_samples=400):  # max_samples=100
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
        if tensor.max() == tensor.min():
            return torch.zeros_like(tensor)
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def setup_data(root_dir, batch_size=4):
    dataset = PowerDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, loss_fn, optimizer, num_epochs, device, save_path):
    model.to(device)
    model.train()
    # Build lr scheduler
    cosine_lr = CosineRestartLr(0.002, [100], [1], 1e-7)  # learning_rate, max_iters, [1], min_learning_rate
    cosine_lr.set_init_lr(optimizer)

    print_freq = 100

    for epoch in range(num_epochs):
        epoch_loss = 0
        total_iterations = len(dataloader)
        iteration_loss = 0
        with tqdm(total=total_iterations, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, (features, labels) in enumerate(dataloader, 1):
                features, labels = features.to(device), labels.to(device)
                # lr scheduler
                regular_lr = cosine_lr.get_regular_lr(epoch)
                cosine_lr._set_lr(optimizer, regular_lr)

                outputs = model(features)
                optimizer.zero_grad()
                outputs = outputs.squeeze(1)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()*100
                iteration_loss += loss.item()*100
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if i % print_freq == 0 or i == total_iterations:
                    print(f"Iteration {i}/{total_iterations}, Iteration Average Loss: {iteration_loss/print_freq:.4f}")
                    iteration_loss = 0

                # Collect gradients
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone().cpu().numpy()
                for name, grad in gradients.items():
                    print(f"Epoch {epoch}, Layer {name}: Mean gradient = {np.mean(grad)}")
        
        avg_loss = epoch_loss / total_iterations
        print(f"Epoch {epoch+1}/{num_epochs}, Epoch Loss: {epoch_loss:.4f} Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 2 == 0:
            # torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), "full_model.pth")

def unfreeze_layers(model, num_layers_to_unfreeze):
    for i, (name, param) in enumerate(model.backbone.named_parameters()): # length 327
        if i >= len(list(model.backbone.parameters())) - num_layers_to_unfreeze:
            param.requires_grad = True

def main():
    # Hyperparameters
    root_dir = './CircuitNet-N28/IR_drop_features_decompressed/'
    batch_size = 4
    num_epochs = 50
    learning_rate = 0.002
    save_path = './checkpoints'
    arg_dict = {'task': 'irdrop_mavi', 'save_path': 'work_dir/irdrop_mavi/', 'pretrained': None, 'max_iters': 1, 'plot_roc': False, 'arg_file': None, 'cpu': True, 'dataroot': 'CircuitNet-N28/training_set/IR_drop', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'IRDropDataset', 'batch_size': 2, 'model_type': 'MAVI', 'in_channels': 1, 'out_channels': 4, 'lr': 0.0002, 'weight_decay': 0.01, 'loss_type': 'L1Loss', 'eval_metric': ['NRMS', 'SSIM'], 'threshold': 0.9885, 'ann_file': './files/train_N28.csv', 'test_mode': False}
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = setup_data(root_dir, batch_size)
    model_name = 'swin_base_patch4_window7_224'
    # model_name = 'swin_base_patch4_window7_224.ms_in22k'
    model = init_model(model_name, input_channels=4, num_classes=0, pretrained=True)
    # model = SimpleModel(input_channels=4) 

    # loss_fn = nn.L1Loss()
    loss_fn = build_loss(arg_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # try freeze layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    # unfreeze_layers(model, 200)


    # Training
    train_model(model, dataloader, loss_fn, optimizer, num_epochs, device, save_path)
    
    # Save final model
    torch.save(model.state_dict(), f"{save_path}/final_model.pth")

if __name__ == "__main__":
    main()