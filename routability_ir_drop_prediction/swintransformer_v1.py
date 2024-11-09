import torch
from torch import nn
from timm import create_model

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_index):
        super(UNetDecoderBlock, self).__init__()
        self.block_index = block_index
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.transconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Input: [16, 1024, 7, 7]
        x = self.conv1(x)
        # [16, 512, 7, 7]
        x = self.relu(x)
        x = self.conv2(x)
        # [16, 512, 7, 7]
        x = self.relu(x)
        x = self.transconv(x)
        # [16, 512, 14, 14]
        return x

class SwinTranformerIR(nn.Module):
    def __init__(self, backbone, input_channels):
        super(SwinTranformerIR, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.backbone = backbone
        self.decoder = nn.Sequential(
            UNetDecoderBlock(1024, 512, 1), # Input: [16, 1024, 7, 7] Output: [16, 512, 14, 14]
            UNetDecoderBlock(512, 256, 2),  # Input: [16, 512, 14, 14] Output: [16, 256, 28, 28]
            UNetDecoderBlock(256, 128, 3),  # Input: [16, 256, 28, 28] Output: [16, 128, 56, 56]
            UNetDecoderBlock(128, 64, 4),  # Input: [16, 128, 56, 56] Output: [16, 64, 112, 112]
            UNetDecoderBlock(64, 32, 5),  # Input: [16, 64, 112, 112] Output: [16, 32, 224, 224]
            nn.Conv2d(32, 1, kernel_size=1), # Input: [16, 32, 224, 224]Output: [16, 1, 224, 224]
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        # Input shape: torch.Size([16, 4, 224, 224])
        
        # Initial convolution
        x = self.initial_conv(x)  # [16, 4, 224, 224] ==> [16, 3, 224, 224]
        
        # Backbone features
        features = self.backbone.forward_features(x) # [16, 3, 224, 224] ==> [16, 7, 7, 1024]
        
        # Permute for decoder
        features = features.permute(0, 3, 1, 2) # [16, 7, 7, 1024] ==> [16, 1024, 7, 7]
        
        # Decoder
        x = self.decoder(features)

        return x     # x.shape = [16, 1, 224, 224]

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