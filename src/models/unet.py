import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.begin = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        self.downsample = nn.ModuleList([
            self._make_downsample_block(64, 128),
            self._make_downsample_block(128, 256),
            self._make_downsample_block(256, 512)
        ])
        
        self.bottle_neck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3)
        )
        
        self.upsample = nn.ModuleList([
            self._make_upsample_block(1024, 512, 256),
            self._make_upsample_block(512, 256, 128),
            self._make_upsample_block(256, 128, 64)
        ])
        
        self.out = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        

    def _make_downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )
    
    def _make_upsample_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3),
        )
            
    def _central_crop(self, X, target_size):
        _, _, h, w = X.shape
        target_h, target_w = target_size
        
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        
        return X[:, :, start_h:start_h+target_h, start_w:start_w+target_w]
    
    
    def forward(self, X):
        # X.shape = (B, 3, 572, 572)
        
        cache = []
        X = self.begin(X)
        cache.append(X)
        
        for block in self.downsample:
            X = block(X)
            cache.append(X)
        
        X = self.bottle_neck(X)
        
        for block in self.upsample:
            X_cache = cache.pop()
            X_cache = self._central_crop(X_cache, X.shape[2:])
            X = torch.cat((X_cache, X), dim=1)
            X = block(X)
        
        X = torch.cat((cache.pop(), X), dim=1)
        X = self.out(X)
        return X