# SwinTransformer backbone for IR-drop 


## 1. Model Architecture

### 1.1 SwinTransformer Backbone
* **Configuration**:
  - Initial convolution layer: 4 → 3 channels 
  - Pretrained `swin_base_patch4_window7_224` as backbone for feature extraction
  - Patch size: 4x4
  - Window size: 7x7

### 1.2 Decoder Structure
* **Architecture**: UNet-style decoder with 5 sequential blocks
* **Decoder Block Components**:
  ```
  For each block:
  - Conv2d(in_channels, out_channels, kernel=3, padding=1)
  - ReLU
  - Conv2d(out_channels, out_channels, kernel=3, padding=1)
  - ReLU
  - ConvTranspose2d(out_channels, out_channels, kernel=2, stride=2)
  ```
* **Channel Progression**:
  ```
  1024 → 512 → 256 → 128 → 64 → 32 → 1
  ```
* **Final Layers**:
  - 1x1 convolution
  - Sigmoid activation

## 2. Data Preprocessing

### 2.1 Input Features
* **Power Features**:
  ```python
  feature_dirs = [
      'power_i',
      'power_s',
      'power_sca',
      'Power_all'
  ]
  ```

### 2.2 Processing Pipeline
```python
# Original size: [259, 259]
# Target size:  [224, 224]

1. Load numpy array → [259, 259]
2. Interpolate: F.interpolate(
       input,
       size=(224, 224),
       mode='bilinear',
       align_corners=True
   )
3. Normalize: (x - min) / (max - min)
4. Stack features: torch.stack(features, dim=0)
   Result: [4, 224, 224]
5. Batch processing(batch size = 16): [16, 4, 224, 224]
```

### 2.3 Label Processing
```python
1. Load IR drop values
2. Interpolate to [224, 224]
3. label = label.clamp(1e-6, 50)
4. label = (torch.log10(label) + 6) / (np.log10(50) + 6)
```

## 3. Input/Output size

### Input Size
```python
[B, C, H, W] = [16, 4, 224, 224]
where:
- B: Batch size (16)
- C: Channel count (4 power features)
- H, W: Height, Width (224x224)
```

### Output Size
```python
[B, C, H, W] = [16, 1, 224, 224]
where:
- B: Batch size (16)
- C: Channel count (1 IR drop prediction)
- H, W: Height, Width (224x224)
```

## 4. Learning Strategy

### 4.1 Hyperparameters
```python
config = {
    'batch_size': 16,
    'learning_rate': 2e-3,
    'weight_decay': 1e-4,
    'accuracy_threshold': 0.1
}
```

### 4.2 Optimizer
```python
optimizer = AdamW([
    {'params': backbone_params, 'lr': learning_rate * 0.1},
    {'params': decoder_params, 'lr': learning_rate}
], weight_decay=1e-4)
```

### 4.3 Learning Rate Schedule
```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=[learning_rate * 0.1, learning_rate],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1e4
)
```

## 5. Evaluation Metrics

### 5.1 Accuracy Calculation

* Accuracy: within +/- 10% error bar--> accurate

### 5.2 Loss Function
* Primary: L1 Loss
* Monitoring metrics:
  - Training/Validation loss
  - Training/Validation accuracy
  - Learning rates (backbone and decoder)

## 6. Sample Input/Output

[Reserved space for visualizations]

### 6.1 Planned Visualizations
1. Input power features (4 channels)
2. Predicted IR drop maps
3. Ground truth IR drop maps
4. Error distribution heatmaps
5. Comparison between predicted and actual values

