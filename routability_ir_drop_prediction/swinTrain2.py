import os
import numpy as np
import json
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
    def __init__(self, root_dir, target_size=(224, 224), max_samples=400, train=True):
        self.root_dir = root_dir
        self.feature_dirs = ['power_i', 'power_s', 'power_sca', 'Power_all']
        self.label_dir = 'IR_drop'
        self.target_size = target_size
        self.train = train
        self.data = []
        
        all_files = sorted(os.listdir(os.path.join(root_dir, self.feature_dirs[0])))
        # Split data into train/val
        split_idx = int(len(all_files) * 0.8)  # 80% train, 20% val
        files_to_use = all_files[:split_idx] if train else all_files[split_idx:]
        
        for i, case_name in enumerate(files_to_use):
            if i >= max_samples:
                break
                
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
        
        for fp in feature_paths:
            feature = np.load(fp)
            feature = torch.tensor(feature, dtype=torch.float32)
            feature = F.interpolate(feature.unsqueeze(0).unsqueeze(0), 
                                  size=self.target_size, 
                                  mode='bilinear', 
                                  align_corners=True).squeeze()
            features.append(self.normalize(feature))
        features = torch.stack(features, dim=0)
        
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.float32)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0), 
                            size=self.target_size, 
                            mode='bilinear', 
                            align_corners=True).squeeze()
        label = label.clamp(1e-6, 50)
        label = (torch.log10(label) + 6) / (np.log10(50) + 6)
        
        if self.train:
            # Apply data augmentation for training
            if torch.rand(1) < 0.5:
                features = torch.flip(features, dims=[-1])
                label = torch.flip(label, dims=[-1])
            if torch.rand(1) < 0.5:
                features = torch.flip(features, dims=[-2])
                label = torch.flip(label, dims=[-2])
        
        return features, label
    
    @staticmethod
    def normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val == min_val:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)

def calculate_accuracy(pred, target, threshold=0.1):
    relative_error = torch.abs(pred - target) / (target + 1e-8)
    correct = (relative_error <= threshold).float().mean()
    return correct * 100.0

def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        self.setup_training()

    def setup_training(self):
        # Separate backbone and decoder parameters
        backbone_params = []
        decoder_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        # Create optimizer with parameter groups
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.1},
            {'params': decoder_params, 'lr': self.config['learning_rate']}
        ], weight_decay=self.config['weight_decay'])

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.config['learning_rate'] * 0.1, self.config['learning_rate']],
            steps_per_epoch=self.config['steps_per_epoch'],
            epochs=self.config['epochs'],
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )

        self.criterion = nn.L1Loss()
        self.scaler = GradScaler()

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]}')
        
        for i, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Mixed precision training
            with autocast():
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(1), labels)
                accuracy = calculate_accuracy(
                    outputs.squeeze(1),
                    labels,
                    threshold=self.config['accuracy_threshold']
                )

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Update metrics
            losses.update(loss.item(), features.size(0))
            accuracies.update(accuracy.item(), features.size(0))

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.2f}%',
                'LR-backbone': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'LR-decoder': f'{self.optimizer.param_groups[1]["lr"]:.6f}'
            })

        return losses.avg, accuracies.avg

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(1), labels)
            accuracy = calculate_accuracy(
                outputs.squeeze(1),
                labels,
                threshold=self.config['accuracy_threshold']
            )

            losses.update(loss.item(), features.size(0))
            accuracies.update(accuracy.item(), features.size(0))

        return losses.avg, accuracies.avg

def train():
    # Configuration
    config = {
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 2e-3,
        'weight_decay': 1e-4,
        'dataroot': './CircuitNet-N28/IR_drop_features_decompressed/',
        'save_path': './checkpoints',
        'print_freq': 10,
        'save_freq': 5,
        'accuracy_threshold': 0.1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # Create save directory
    os.makedirs(config['save_path'], exist_ok=True)
    
    # Initialize datasets and dataloaders
    train_dataset = PowerDataset(config['dataroot'], train=True)
    val_dataset = PowerDataset(config['dataroot'], train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Update config with steps per epoch
    config['steps_per_epoch'] = len(train_loader)
    
    # Initialize model
    model = init_model(
        'swin_base_patch4_window7_224',
        input_channels=4,
        num_classes=0,
        pretrained=True
    ).to(config['device'])
    
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Training loop
    best_val_accuracy = 0
    for epoch in range(config['epochs']):
        # Training phase
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # Validation phase
        val_loss, val_acc = trainer.validate(val_loader)
        
        print(f'Epoch {epoch + 1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoints
        if (epoch + 1) % config['save_freq'] == 0:
            save_checkpoint(
                model,
                trainer.optimizer,
                trainer.scheduler,
                epoch,
                val_loss,
                val_acc,
                os.path.join(config['save_path'], f'checkpoint_epoch_{epoch + 1}.pth')
            )
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_checkpoint(
                model,
                trainer.optimizer,
                trainer.scheduler,
                epoch,
                val_loss,
                val_acc,
                os.path.join(config['save_path'], 'best_model.pth')
            )
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train()