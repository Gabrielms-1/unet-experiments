import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels=in_channels, out_channels=64)
        self.enc2 = self.conv_block(in_channels=64, out_channels=128)
        self.enc3 = self.conv_block(in_channels=128, out_channels=256)
        self.enc4 = self.conv_block(in_channels=256, out_channels=512)

        # Bottleneck
        self.bottleneck = self.conv_block(in_channels=512, out_channels=1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.deconv4 = self.conv_block(in_channels=1024, out_channels=512)
        
        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.deconv3 = self.conv_block(in_channels=512, out_channels=256)
        
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.deconv2 = self.conv_block(in_channels=256, out_channels=128)
        
        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.deconv1 = self.conv_block(in_channels=128, out_channels=64)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)


    def conv_block(self, in_channels, out_channels):
        """Bloco de convolução: 2 convoluções + BatchNorm + ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )    
    
    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(e4, kernel_size=2))
        # Decoder
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.deconv4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.deconv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.deconv2(d2)
        print(f"Shape da saída do deconv2 (d2): {d2.shape}")

        d1 = self.upconv1(d2)
        print(f"Shape da saída do upconv1 (d1): {d1.shape}")
        d1 = torch.cat((e1, d1), dim=1)
        print(f"Shape da saída do cat (d1): {d1.shape}")
        d1 = self.deconv1(d1)
        print(f"Shape da saída do deconv1 (d1): {d1.shape}")

        return self.final_conv(d1)
        

        


