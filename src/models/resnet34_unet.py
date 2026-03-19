from abc import abstractmethod
from typing import Type
import torch
import torch.nn as nn

class UpsampleResidualStrategy(nn.Module): 
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    @abstractmethod
    def getResidual(self, X) -> torch.Tensor:
        pass
    
class ZeroPaddingResidualStrategy(UpsampleResidualStrategy):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride)
        
    def forward(self, X):
        _, _, h, w = X.shape
        
        if self.in_channels == self.out_channels:
            return X
        
        padding_channels = self.out_channels - self.in_channels
        padding = torch.zeros(X.shape[0], padding_channels, h, w, device=X.device)
        X = torch.cat((X, padding), dim=1)
        
        return X
    
class AllProjectStrategy(UpsampleResidualStrategy):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride)
        
        self.trans = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(self.out_channels)
        )
        
    def forward(self, X):   
        return self.trans(X)
    
class UpsampleProjectStrategy(UpsampleResidualStrategy):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride)
        
        self.trans = nn.Identity()
        if self.in_channels != self.out_channels:
            self.trans = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, X):
        return self.trans(X)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strategy: Type[UpsampleResidualStrategy], kernel_size=3, stride1=1, stride2=1, padding=1):
        super().__init__()
        self.residual = strategy(in_channels, out_channels, stride1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride1, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride2, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, X):
        shortcut = self.residual(X)
        
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        
        X = self.conv2(X)
        X = self.bn2(X)
        
        X += shortcut
        X = self.relu2(X)
        
        return X
        
    
class Resnet34_Unet(nn.Module):
    def __init__(self, strategy: Type[UpsampleResidualStrategy]):
        super().__init__()
        
        self.encoder1 = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        self.encoder2 = nn.ModuleList([
            ResidualBlock(
                64, 64, strategy, kernel_size=3
            ) for i in range(3)
        ])
        
        self.encoder3 = nn.ModuleList([
            ResidualBlock(
                64, 128, strategy, kernel_size=3, stride1=2
            ),
            *[
                ResidualBlock(
                    128, 128, strategy, kernel_size=3
                ) for i in range(3)
            ]
        ])
        
        self.encoder4 = nn.ModuleList([
            ResidualBlock(
                128, 256, strategy, kernel_size=3, stride1=2
            ),
            *[
                ResidualBlock(
                    256, 256, strategy, kernel_size=3
                ) for i in range(5)
            ]
        ])
        
        self.encoder5 = nn.ModuleList([
            ResidualBlock(
                256, 512, strategy, kernel_size=3, stride1=2
            ),
            *[
                ResidualBlock(
                    512, 512, strategy, kernel_size=3
                ) for i in range(2)
            ]
        ])
        
        # Not sure
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.ModuleList([
            nn.Sequential(
                self._make_upsample_block(256 + 512, 32, 32),
                self._make_upsample_block(32, 32, 32),
            ),
            nn.Sequential(
                self._make_upsample_block(32 + 256, 32, 32),
                self._make_upsample_block(32, 32, 32)
            ),
            nn.Sequential(
                self._make_upsample_block(32 + 128, 32, 32),
                self._make_upsample_block(32, 32, 32)
            ),
            nn.Sequential(
                self._make_upsample_block(32 + 64, 32, 32),
                self._make_upsample_block(32, 32, 32)
            ),
            nn.Sequential(
                self._make_upsample_block(32, 32, 32),
                self._make_upsample_block(32, 32, 32)
            )
        ])
        
        self.out = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3)
        )
        
    def _make_upsample_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def _central_crop(self, X, target_size):
        _, _, h, w = X.shape
        target_h, target_w = target_size
        
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        
        return X[:, :, start_h:start_h+target_h, start_w:start_w+target_w]
    
    def forward(self, X):
        # X.shape = (B, 3, 768, 544) 
        cache = []
        residual = None
        
        for block in self.encoder1:
            X = block(X)
        
        for i, block in enumerate(self.encoder2):
            X = block(X)
            
            if i == len(self.encoder2) - 1:
                cache.append(X)
        
        for i, block in enumerate(self.encoder3):
            X = block(X)
            
            if i == len(self.encoder3) - 1:
                cache.append(X)
        
        for i, block in enumerate(self.encoder4):
            X = block(X)
            
            if i == len(self.encoder4) - 1:
                cache.append(X)
        
        for i, block in enumerate(self.encoder5):
            X = block(X)
            
            if i == len(self.encoder5) - 1:
                cache.append(X)
        
        X = self.bottleneck(X)
        
        for i, block in enumerate(self.decoder1):
            if i != 4:
                X_cache = cache.pop()
                X_cache = self._central_crop(X_cache, X.shape[2:])
                X = torch.cat((X_cache, X), dim=1)
            X = block(X)
            
        X = self.out(X)
        
        return X