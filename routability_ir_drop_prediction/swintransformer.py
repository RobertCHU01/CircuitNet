import torch
from torch import nn
import torch.nn.functional as F
from timm import create_model

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        return self.up(self.conv(x))

class SwinTransformerIRMavi(nn.Module):
    def __init__(self, backbone, input_channels=1, output_channels=4):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 3D processing before Swin
        self.initial_3d = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Reduce to 2D for Swin
        self.to_2d = nn.Conv2d(32, 3, kernel_size=1)
        self.backbone = backbone
        
        # Get the backbone output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.backbone.forward_features(dummy_input)
            self.backbone_channels = dummy_output.shape[-1]
            
        # Post-backbone processing
        self.post_backbone = nn.Sequential(
            nn.Conv2d(self.backbone_channels, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path
        self.decoder = nn.ModuleList([
            UNetDecoderBlock(1024, 512),
            UNetDecoderBlock(512, 256),
            UNetDecoderBlock(256, 128),
            UNetDecoderBlock(128, 64),
            UNetDecoderBlock(64, 32)
        ])
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input shape debugging
        # print(f"Input shape: {x.shape}")
        
        # Store input for later use (B, 1, D, H, W)
        x_in = x
        
        # Initial 3D processing
        x = self.initial_3d(x)
        # print(f"After 3D processing: {x.shape}")
        
        # Convert to 2D by taking mean across depth dimension
        x = x.mean(dim=2)
        # print(f"After depth reduction: {x.shape}")
        
        x = self.to_2d(x)
        # print(f"After to_2d: {x.shape}")
        
        # Get features from Swin backbone
        features = self.backbone.forward_features(x)
        # print(f"Backbone output shape: {features.shape}")
        
        # Reshape features from [B, H, W, C] to [B, C, H, W]
        if len(features.shape) == 4:  # [B, H, W, C]
            features = features.permute(0, 3, 1, 2)
        elif len(features.shape) == 3:  # [B, L, C]
            B, L, C = features.shape
            H = W = int((L ** 0.5))
            features = features.transpose(1, 2).reshape(B, C, H, W)
            
        # print(f"After permute: {features.shape}")
        
        # Post-backbone processing
        features = self.post_backbone(features)
        # print(f"After post-backbone: {features.shape}")
        
        # Decoder path
        decoder_outputs = []
        for i, decoder_block in enumerate(self.decoder):
            features = decoder_block(features)
            decoder_outputs.append(features)
            # print(f"After decoder block {i+1}: {features.shape}")
        
        # Final output processing
        logits = self.final(features)
        # print(f"After final layer: {logits.shape}")
        
        # Match dimensions with input
        if x_in.shape[-2:] != logits.shape[-2:]:
            logits = F.interpolate(logits, size=x_in.shape[-2:], 
                                 mode='bilinear', align_corners=True)
            # print(f"After interpolation: {logits.shape}")
        
        # Final processing
        logits = logits * x_in.squeeze(1)
        output = torch.sum(logits, dim=1, keepdim=True)
        # print(f"Final output shape: {output.shape}")
        
        return output

def init_model(model_name, input_channels=1, output_channels=4, pretrained=True):
    swin_transformer = create_model(model_name, pretrained=pretrained, num_classes=0)
    model = SwinTransformerIRMavi(swin_transformer, input_channels, output_channels)
    return model