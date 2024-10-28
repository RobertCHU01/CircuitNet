import torch
from torch import nn
from timm import create_model

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.transconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.transconv(x)
        return x

class SwinTranformerIR(nn.Module):
    def __init__(self, backbone, input_channels):
        super(SwinTranformerIR, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.backbone = backbone
        self.decoder = nn.Sequential(
            UNetDecoderBlock(1024, 512),
            UNetDecoderBlock(512, 256),
            UNetDecoderBlock(256, 128),
            UNetDecoderBlock(128, 64),
            UNetDecoderBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),  # Final layer to get the desired output channels
            # nn.ReLU()  # Normalize output to range [0, 1]
            nn.Sigmoid()  # Changed to Sigmoid for [0, 1] output
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.initial_conv(x)  # Reduce input channels to 3
        # Get the backbone features
        features = self.backbone.forward_features(x)  # Shape: [batch_size, 7, 7, 1024]

        # Permute to match Conv2d input shape: [batch_size, channels, height, width]
        features = features.permute(0, 3, 1, 2)  # Shape: [batch_size, 1024, 7, 7]

        # Apply decoder
        x = self.decoder(features)
        return x

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                if 'backbone' not in name:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                if 'backbone' not in name:
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

def init_model(model_name, input_channels=3, num_classes=0, pretrained=True):
    swin_transformer = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = SwinTranformerIR(swin_transformer, input_channels)
    
    return model