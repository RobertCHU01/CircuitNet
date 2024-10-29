import torch
from torch import nn
from timm import create_model

class TemporalSwinTransformer(nn.Module):
    def __init__(self, backbone, input_channels=4):
        super().__init__()
        
        # Input adaptation layer
        self.input_adapt = nn.Conv2d(input_channels, 3, kernel_size=1)
        
        # Swin backbone
        self.backbone = backbone
        
        # Feature dimension from Swin-B
        swin_dims = 1024  # This is the output dimension of Swin-B
        
        # Decoder path with corrected dimensions
        self.decoder = nn.Sequential(
            # First upsampling block
            nn.Conv2d(swin_dims, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Second upsampling block
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Third upsampling block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Fourth upsampling block
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Final refinement
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        self._initialize_weights()

    def forward(self, x):
        # Input shape: [batch, channels, height, width]
        input_size = x.shape[-2:]  # Store original input size
        
        # Adapt input channels for Swin
        x = self.input_adapt(x)
        
        # Get features from Swin backbone
        features = self.backbone.forward_features(x)
        
        # Reshape features from backbone
        features = features.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        
        # Decode and ensure output size matches input size
        x = self.decoder(features)
        
        # Final resize to match target size if needed
        if x.shape[-2:] != input_size:
            x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def init_model(model_name, input_channels=4, num_classes=0, pretrained=True):
    swin_transformer = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = TemporalSwinTransformer(swin_transformer, input_channels)
    return model