import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    """Feature Pyramid Network"""
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # Add BatchNorm and ReLU
        for in_channels in in_channels_list:
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            
    def forward(self, features):
        """
        Args:
            features: List of bottom-up feature maps, arranged from shallow to deep
        Returns:
            results: List of enhanced multi-scale feature maps
        """
        # Check input
        if len(features) != len(self.lateral_convs):
            raise ValueError(f'Expected {len(self.lateral_convs)} features, but got {len(features)}')
        
        # Transform feature dimensions
        laterals = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            # Ensure feature map dimensions are correct
            if feature.dim() != 4:
                raise ValueError(f'Feature {i} expected 4D tensor, but got {feature.dim()}D')
            if feature.size(1) != lateral_conv[0].in_channels:
                raise ValueError(f'Feature {i} expected {lateral_conv[0].in_channels} channels, but got {feature.size(1)}')
            laterals.append(lateral_conv(feature))
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            # Check feature map size before upsampling
            if laterals[i].shape[-2:] != laterals[i-1].shape[-2:]:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i],
                    size=laterals[i - 1].shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
            else:
                laterals[i - 1] = laterals[i - 1] + laterals[i]
            
        # Final convolution
        results = []
        for lateral, fpn_conv in zip(laterals, self.fpn_convs):
            results.append(fpn_conv(lateral))
        
        return results

class FeatureFusion(nn.Module):
    """Feature fusion module"""
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        # Ensure feature map sizes are consistent
        if x1.shape[-2:] != x2.shape[-2:]:
            x2 = F.interpolate(
                x2,
                size=x1.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Concatenate and fuse features
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x 