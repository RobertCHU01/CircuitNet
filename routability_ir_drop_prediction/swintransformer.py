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
        self.initial_conv = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.backbone = backbone
        self.decoder = nn.Sequential(
            UNetDecoderBlock(1024, 512),
            UNetDecoderBlock(512, 256),
            UNetDecoderBlock(256, 128),
            UNetDecoderBlock(128, 64),
            UNetDecoderBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.initial_conv(x)
        features = self.backbone.forward_features(x)
        features = features.permute(0, 3, 1, 2)
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