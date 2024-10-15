import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np

class PowerDataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224)):
        self.root_dir = root_dir
        self.feature_dirs = ['power_i', 'power_s', 'power_sca', 'Power_all']
        self.label_dir = 'IR_drop'
        self.target_size = target_size
        self.data = []
        for case_name in os.listdir(os.path.join(root_dir, self.feature_dirs[0]))[:100]:  # Limit to 100 samples
            feature_paths = [os.path.join(root_dir, feature_dir, case_name) for feature_dir in self.feature_dirs]
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
            feature = F.interpolate(feature.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='nearest').squeeze()
            features.append((feature - feature.min()) / (feature.max() - feature.min() + 1e-8))
        features = torch.stack(features, dim=0)
        
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.float32)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='nearest').squeeze()
        label = label.clamp(1e-6, 50)
        label = (torch.log10(label) + 6) / (np.log10(50) + 6)
        
        return features, label

class SimpleModel(nn.Module):
    def __init__(self, input_channels=4):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x.squeeze(1)

def train_model(model, dataloader, num_epochs=10, lr=0.001):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    return model

# Main execution
root_dir = './CircuitNet-N28/IR_drop_features_decompressed/'
dataset = PowerDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SimpleModel(input_channels=4)
trained_model = train_model(model, dataloader)

# Save the model
torch.save(trained_model.state_dict(), 'simple_model.pth')