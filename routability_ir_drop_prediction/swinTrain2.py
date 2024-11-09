import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from swintransformer import init_model
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PowerDataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224), max_samples=None, train=True):
        self.root_dir = root_dir
        self.feature_dirs = ['power_i', 'power_s', 'power_sca', 'Power_all']
        self.label_dir = 'IR_drop'
        self.target_size = target_size
        self.train = train
        self.data = []
        
        all_files = sorted(os.listdir(os.path.join(root_dir, self.feature_dirs[0])))
        if max_samples:
            if train:
                files_to_use = all_files[:int(0.9 * max_samples)]
            else:
                files_to_use = all_files[int(0.9 * max_samples):max_samples]
        else:
            files_to_use = all_files
        
        for case_name in files_to_use:
            feature_paths = [os.path.join(root_dir, feature_dir, case_name) 
                           for feature_dir in self.feature_dirs]
            label_path = os.path.join(root_dir, self.label_dir, case_name)
            if all(os.path.exists(fp) for fp in feature_paths) and os.path.exists(label_path):
                self.data.append((feature_paths, label_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_paths, label_path = self.data[idx]
        features = []
        
        # Load and process features
        for fp in feature_paths:
            feature = np.load(fp)
            feature = torch.tensor(feature, dtype=torch.float32)
            feature = F.interpolate(feature.unsqueeze(0).unsqueeze(0), 
                                  size=self.target_size, 
                                  mode='bilinear', 
                                  align_corners=True).squeeze()
            features.append(self.normalize(feature))
        
        # Stack features with correct dimensions: (1, D, H, W)
        features = torch.stack(features, dim=0)  # D, H, W
        features = features.unsqueeze(0)  # Add channel dimension: 1, D, H, W
        
        # Load and process label
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.float32)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0), 
                            size=self.target_size, 
                            mode='bilinear', 
                            align_corners=True)
        label = label.clamp(1e-6, 50)
        label = (torch.log10(label) + 6) / (np.log10(50) + 6)
        
        return features, label.squeeze(0)  # Return (1, D, H, W), (1, H, W)
    
    @staticmethod
    def normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val == min_val:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # Ensure shapes match
        if pred.shape != target.shape:
            pred = F.interpolate(pred.unsqueeze(1), size=target.shape[-2:], 
                               mode='bilinear', align_corners=True).squeeze(1)
        
        # Combine L1 and MSE losses
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return 0.5 * l1 + 0.5 * mse

def train_model(config):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize datasets and dataloaders
    train_dataset = PowerDataset(config['dataroot'], max_samples=100, train=True)
    val_dataset = PowerDataset(config['dataroot'], max_samples=100, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = init_model('swin_base_patch4_window7_224', 
                      input_channels=1, output_channels=4, 
                      pretrained=True).to(device)
    
    # Set up optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['lr_backbone']},
        {'params': model.initial_3d.parameters(), 'lr': config['lr_decoder']},
        {'params': model.decoder.parameters(), 'lr': config['lr_decoder']},
        {'params': model.final.parameters(), 'lr': config['lr_decoder']}
    ], weight_decay=config['weight_decay'])
    
    # Set up loss and scaler for mixed precision training
    criterion = CustomLoss()
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = AverageMeter()
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{config["epochs"]}') as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss.update(loss.item(), features.size(0))
                pbar.set_postfix({'train_loss': f'{train_loss.avg:.4f}'})
                pbar.update()
        
        # Validation phase
        model.eval()
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss.update(loss.item(), features.size(0))
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss.avg:.4f}, Val Loss: {val_loss.avg:.4f}')
        
        # Save best model
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss.avg,
            }, 'best_model.pth')

if __name__ == "__main__":
    config = {
        'dataroot': './CircuitNet-N28/IR_drop_features_decompressed/',
        'batch_size': 8,
        'epochs': 100,
        'lr_backbone': 1e-5,
        'lr_decoder': 2e-4,
        'weight_decay': 1e-4,
    }
    train_model(config)