import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterWeightedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, sigma=1.0, center_weight_power=0.5):
        super(CenterWeightedSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.sigma = sigma
        self.center_weight_power = center_weight_power
        self.register_buffer('center_weight', None)

    def create_center_weight(self, size):
        if self.center_weight is not None and self.center_weight.size() == (1, 1, size[2], size[3]):
            return self.center_weight

        center_x = size[2] // 2
        center_y = size[3] // 2

        x = torch.arange(size[2], device=self.conv.weight.device).float()
        y = torch.arange(size[3], device=self.conv.weight.device).float()

        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')

        distance = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_distance = torch.sqrt(torch.tensor(center_x**2 + center_y**2).float())

        weight = torch.exp(-distance**2 / (2 * self.sigma**2))

        self.register_buffer('center_weight', weight.unsqueeze(0).unsqueeze(0))
        return self.center_weight

    def forward(self, x):

        center_weight = self.create_center_weight(x.size())

        enhanced_center_weight = center_weight ** self.center_weight_power

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        attention = self.sigmoid(self.conv(x_cat))

        attention = attention * enhanced_center_weight

        return x * attention.expand_as(x)

class EdgeSuppressionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(EdgeSuppressionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.center_attention = CenterWeightedSpatialAttention()

    def forward(self, x):

        edge_features = self.edge_conv(x)

        center_weighted = self.center_attention(x)

        edge_weight = torch.sigmoid(edge_features) * 0.2
        return center_weighted * (1 - edge_weight) 

class AdaptiveTextureAttention(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveTextureAttention, self).__init__()

        self.texture_scales = [3, 5, 7, 9, 11]
        self.texture_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels//4, k, padding=k//2, groups=in_channels//4),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//4, in_channels//4, 1),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True)
            ) for k in self.texture_scales
        ])

        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(self.texture_scales), 1),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(5*in_channels//4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        texture_feats = [branch(x) for branch in self.texture_branches]

        weights = self.weight_generator(x)

        weighted_feats = []
        for i, feat in enumerate(texture_feats):
            weighted_feats.append(feat * weights[:, i:i+1])

        fused_texture = torch.cat(weighted_feats, dim=1)
        return self.fusion(fused_texture)

class DynamicSpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 7, 11, 15]):
        super(DynamicSpatialAttention, self).__init__()

        self.spatial_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 8, k, padding=k//2),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])

        self.attention_generator = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * 8, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        self.local_global_fusion = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)

        multi_scale_feats = []
        for branch in self.spatial_branches:
            feat = branch(spatial_features)
            multi_scale_feats.append(feat)

        fused_spatial = torch.cat(multi_scale_feats, dim=1)
        attention = self.attention_generator(fused_spatial)

        local_global = self.local_global_fusion(spatial_features)

        final_attention = attention * local_global

        return x * final_attention

class EnhancedRockAttention(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedRockAttention, self).__init__()

        self.texture_attention = AdaptiveTextureAttention(in_channels)

        self.spatial_attention = DynamicSpatialAttention()

        self.edge_suppression = EdgeSuppressionModule(in_channels)

        self.recalibration = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        texture_enhanced = self.texture_attention(x)

        spatial_enhanced = self.spatial_attention(texture_enhanced)

        edge_suppressed = self.edge_suppression(spatial_enhanced)

        recalibrated = self.recalibration(edge_suppressed)

        attention_weight = 0.3
        return x * (1 - attention_weight) + (x * recalibrated) * attention_weight 

class EfficientNetLightAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(EfficientNetLightAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):

        ca = self.channel_attention(x)
        x_ca = x * ca

        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x_sa = x_ca * sa

        return (1 - self.alpha) * x + self.alpha * x_sa

class EfficientNetMinimalAttention(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNetMinimalAttention, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

        self.enhancement_weight = nn.Parameter(torch.tensor(1.1))

    def forward(self, x):

        se_weights = self.se(x)

        enhanced = x * se_weights * self.enhancement_weight

        return enhanced

class EfficientNetRockAttention(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNetRockAttention, self).__init__()

        self.minimal_attention = EfficientNetMinimalAttention(in_channels)

    def forward(self, x):

        return self.minimal_attention(x)

class IdentityAttention(nn.Module):
    def __init__(self, in_channels):
        super(IdentityAttention, self).__init__()

        self.adjustment = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.adjustment
