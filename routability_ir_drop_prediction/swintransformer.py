import torch
from torch import nn
from timm import create_model

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.transconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.transconv(x)
        return x

class SwinTransformerIR(nn.Module):
    def __init__(self, backbone, input_channels):
        super(SwinTransformerIR, self).__init__()
        
        # Initial convolution with BatchNorm and Dropout
        self.initial_conv = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(3, momentum=0.05)
        self.initial_dropout = nn.Dropout(0.1)
        
        self.backbone = backbone

        # Decoder with enhanced blocks
        self.decoder = nn.Sequential(
            UNetDecoderBlock(1024, 512),
            nn.BatchNorm2d(512, momentum=0.05),
            nn.Dropout(0.1),
            
            UNetDecoderBlock(512, 256),
            nn.BatchNorm2d(256, momentum=0.05),
            nn.Dropout(0.1),
            
            UNetDecoderBlock(256, 128),
            nn.BatchNorm2d(128, momentum=0.05),
            nn.Dropout(0.1),
            
            UNetDecoderBlock(128, 64),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.Dropout(0.1),
            
            UNetDecoderBlock(64, 32),
            nn.BatchNorm2d(32, momentum=0.05),
            # No dropout before final layer
            
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        # Apply initial conv with BatchNorm and Dropout
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_dropout(x)
        
        # Get backbone features
        features = self.backbone.forward_features(x)
        features = features.permute(0, 3, 1, 2)
        
        # Apply decoder
        x = self.decoder(features)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def init_model(model_name, input_channels=3, num_classes=0, pretrained=True):
    swin_transformer = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = SwinTransformerIR(swin_transformer, input_channels)
    return model