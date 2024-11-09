import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

class PowerDataset:
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
        _, label_path = self.data[idx]
        
        # Load and process only the label since we're just analyzing the labels
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.float32)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0), 
                            size=self.target_size, 
                            mode='bilinear', 
                            align_corners=True)
        label = label.clamp(1e-6, 50)
        label = (torch.log10(label) + 6) / (np.log10(50) + 6)
        
        return label.squeeze()

def analyze_zero_values():
    # Configuration
    config = {
        'dataroot': './CircuitNet-N28/IR_drop_features_decompressed/',
        'batch_size': 8,
        'target_size': (224, 224),
        'max_samples': 10000  # Limit samples for faster evaluation
    }
    
    # Initialize dataset and dataloader
    dataset = PowerDataset(config['dataroot'], 
                          target_size=config['target_size'],
                          max_samples=config['max_samples'],
                          train=False)
    dataloader = DataLoader(dataset, 
                          batch_size=config['batch_size'],
                          shuffle=False,
                          num_workers=4)
    
    # Initialize counters
    total_zeros = 0
    total_elements = 0
    
    # Tolerance for considering a value as zero (to handle floating point precision)
    epsilon = 1e-6
    
    # Analyze labels
    print("Analyzing label values...")
    with torch.no_grad():
        for labels in tqdm(dataloader):
            # Count zeros in this batch
            zeros = (torch.abs(labels) < epsilon).sum().item()
            total_zeros += zeros
            total_elements += labels.numel()
            
            # Optional: print some sample values from the first batch
            if total_elements == labels.numel():  # First batch
                print("\nSample of actual values from first batch:")
                print(labels[0, :5, :5])  # Print 5x5 sample
    
    # Calculate and print results
    zero_percentage = (total_zeros / total_elements) * 100
    
    print("\nZero Value Analysis Results:")
    print("-" * 50)
    print(f"Total elements analyzed: {total_elements:,}")
    print(f"Number of zero values: {total_zeros:,}")
    print(f"Percentage of zeros: {zero_percentage:.2f}%")
    print("-" * 50)
    print(f"\nNote: Values below {epsilon} are considered zero to account for floating point precision.")

if __name__ == "__main__":
    analyze_zero_values()