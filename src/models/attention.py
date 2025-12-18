import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterWeightedSpatialAttention(nn.Module):
    """Center-weighted spatial attention module"""
    def __init__(self, kernel_size=7, sigma=1.0, center_weight_power=0.5):
        super(CenterWeightedSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.sigma = sigma
        self.center_weight_power = center_weight_power  # Control center weight strength
        self.register_buffer('center_weight', None)
        
    def create_center_weight(self, size):
        """Create center weight matrix"""
        if self.center_weight is not None and self.center_weight.size() == (1, 1, size[2], size[3]):
            return self.center_weight
            
        center_x = size[2] // 2
        center_y = size[3] // 2
        
        # Ensure tensors are created on the correct device
        x = torch.arange(size[2], device=self.conv.weight.device).float()
        y = torch.arange(size[3], device=self.conv.weight.device).float()
        
        # Use indexing='ij' parameter
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        
        # Calculate distance to center
        distance = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_distance = torch.sqrt(torch.tensor(center_x**2 + center_y**2).float())
        
        # Generate Gaussian weights
        weight = torch.exp(-distance**2 / (2 * self.sigma**2))
        
        # Save to buffer
        self.register_buffer('center_weight', weight.unsqueeze(0).unsqueeze(0))
        return self.center_weight

    def forward(self, x):
        # Generate center weight
        center_weight = self.create_center_weight(x.size())
        
        # Enhance center weight influence - use exponential adjustment
        enhanced_center_weight = center_weight ** self.center_weight_power
        
        # Calculate channel attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(x_cat))
        
        # Combine center weight to enhance center region influence
        attention = attention * enhanced_center_weight
        
        return x * attention.expand_as(x)

class EdgeSuppressionModule(nn.Module):
    """Edge suppression module"""
    def __init__(self, channels, reduction=16):
        super(EdgeSuppressionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.center_attention = CenterWeightedSpatialAttention()
        
    def forward(self, x):
        # Extract edge features
        edge_features = self.edge_conv(x)
        
        # Apply center attention
        center_weighted = self.center_attention(x)
        
        # Gentle edge suppression (reduce suppression strength)
        edge_weight = torch.sigmoid(edge_features) * 0.2  # Use sigmoid and lower weight
        return center_weighted * (1 - edge_weight) 


class AdaptiveTextureAttention(nn.Module):
    """Adaptive texture attention module"""
    def __init__(self, in_channels):
        super(AdaptiveTextureAttention, self).__init__()
        
        # Multi-scale texture feature extraction
        self.texture_scales = [3, 5, 7, 9, 11]  # Add finer-grained receptive fields
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
        
        # Adaptive weight generation
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(self.texture_scales), 1),
            nn.Softmax(dim=1)
        )
        
        # Fix: Ensure fusion layer input channel count is correct
        # 5 scales, each outputs in_channels//4 channels, total 5*in_channels//4 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(5*in_channels//4, in_channels, 1),  # Fix channel count
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Extract multi-scale texture features
        texture_feats = [branch(x) for branch in self.texture_branches]
        
        # Generate adaptive weights
        weights = self.weight_generator(x)
        
        # Weighted fusion of texture features
        weighted_feats = []
        for i, feat in enumerate(texture_feats):
            weighted_feats.append(feat * weights[:, i:i+1])
        
        fused_texture = torch.cat(weighted_feats, dim=1)
        return self.fusion(fused_texture)

class DynamicSpatialAttention(nn.Module):
    """Dynamic spatial attention module"""
    def __init__(self, kernel_sizes=[3, 7, 11, 15]):
        super(DynamicSpatialAttention, self).__init__()
        
        self.spatial_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 8, k, padding=k//2),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        
        # Dynamic attention map generation
        self.attention_generator = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * 8, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Local-global feature fusion
        self.local_global_fusion = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract statistical features
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # Multi-scale spatial features
        multi_scale_feats = []
        for branch in self.spatial_branches:
            feat = branch(spatial_features)
            multi_scale_feats.append(feat)
        
        # Fuse multi-scale features
        fused_spatial = torch.cat(multi_scale_feats, dim=1)
        attention = self.attention_generator(fused_spatial)
        
        # Local-global feature fusion
        local_global = self.local_global_fusion(spatial_features)
        
        # Combine attention
        final_attention = attention * local_global
        
        return x * final_attention

class EnhancedRockAttention(nn.Module):
    """Enhanced rock attention module"""
    def __init__(self, in_channels):
        super(EnhancedRockAttention, self).__init__()
        
        # Texture attention
        self.texture_attention = AdaptiveTextureAttention(in_channels)
        
        # Dynamic spatial attention
        self.spatial_attention = DynamicSpatialAttention()
        
        # Add edge suppression module
        self.edge_suppression = EdgeSuppressionModule(in_channels)
        
        # Feature recalibration
        self.recalibration = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Texture enhancement
        texture_enhanced = self.texture_attention(x)
        
        # Spatial attention
        spatial_enhanced = self.spatial_attention(texture_enhanced)
        
        # Gentle edge suppression module
        edge_suppressed = self.edge_suppression(spatial_enhanced)
        
        # Feature recalibration
        recalibrated = self.recalibration(edge_suppressed)
        
        # Add residual connection to prevent over-suppression
        attention_weight = 0.3  # Reduce attention influence strength
        return x * (1 - attention_weight) + (x * recalibrated) * attention_weight 

class EfficientNetLightAttention(nn.Module):
    """Lightweight attention module designed specifically for EfficientNet"""
    def __init__(self, in_channels, reduction=8):
        super(EfficientNetLightAttention, self).__init__()
        
        # Lightweight channel attention (similar to SE module but gentler)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Simplified spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Residual weight control
        self.alpha = nn.Parameter(torch.tensor(0.3))  # Smaller initial weight
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x_sa = x_ca * sa
        
        # Gentle residual connection to avoid over-suppression
        return (1 - self.alpha) * x + self.alpha * x_sa

class EfficientNetMinimalAttention(nn.Module):
    """Minimal attention for EfficientNet - solves over-suppression problem"""
    def __init__(self, in_channels):
        super(EfficientNetMinimalAttention, self).__init__()
        
        # Keep only the most basic SE module, no additional suppression
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Learnable enhancement weight, initialized close to 1 to preserve features
        self.enhancement_weight = nn.Parameter(torch.tensor(1.1))  # Slightly greater than 1, slight enhancement
        
    def forward(self, x):
        # SE attention weights
        se_weights = self.se(x)
        
        # Slight enhancement instead of suppression
        enhanced = x * se_weights * self.enhancement_weight
        
        # Directly return enhanced features without additional suppression
        return enhanced

class EfficientNetRockAttention(nn.Module):
    """Attention module designed specifically for EfficientNet rock classification - fixed version"""
    def __init__(self, in_channels):
        super(EfficientNetRockAttention, self).__init__()
        
        # Use minimal attention instead of complex multi-layer suppression
        self.minimal_attention = EfficientNetMinimalAttention(in_channels)
        
    def forward(self, x):
        # Directly use minimal attention to avoid multiple suppressions
        return self.minimal_attention(x)

class IdentityAttention(nn.Module):
    """Identity attention - for testing, basically does not change input"""
    def __init__(self, in_channels):
        super(IdentityAttention, self).__init__()
        # Only make very small adjustments, close to identity mapping
        self.adjustment = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        return x * self.adjustment

class MultiScaleResidualAttention(nn.Module):
    """Multi-scale residual attention module designed specifically for ResNet50"""
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleResidualAttention, self).__init__()
        
        # Multi-scale feature extraction branches
        self.scales = [1, 3, 5, 7]  # Convolution kernels of different scales
        self.multi_scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=scale, 
                         padding=scale//2, groups=in_channels//4),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ) for scale in self.scales
        ])
        
        # Channel attention - improved version of SE module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention - optimized for rock textures
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection weight parameter
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        residual = x
        
        # Multi-scale feature extraction
        multi_scale_feats = []
        for branch in self.multi_scale_branches:
            feat = branch(x)
            multi_scale_feats.append(feat)
        
        # Fuse multi-scale features
        fused_multi_scale = torch.cat(multi_scale_feats, dim=1)
        
        # Channel attention
        channel_att = self.channel_attention(x)
        channel_refined = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(channel_refined, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_refined, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        spatial_refined = channel_refined * spatial_att
        
        # Feature fusion
        output = self.fusion(spatial_refined)
        
        # Learn residual connection weights
        return self.alpha * output + (1 - self.alpha) * residual

class PyramidPoolingAttention(nn.Module):
    """Pyramid pooling attention module - specifically handles rock textures at different scales"""
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingAttention, self).__init__()
        
        self.pool_sizes = pool_sizes
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        
        # Feature aggregation
        self.aggregation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_feats = []
        
        for pool in self.pools:
            pooled = pool(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_feats.append(upsampled)
        
        # Fuse all scale features
        fused = torch.cat(pyramid_feats, dim=1)
        attention_map = self.aggregation(fused)
        
        return x * attention_map

class ResNet50EnhancedAttention(nn.Module):
    """Enhanced attention module combination designed specifically for ResNet50"""
    def __init__(self, in_channels):
        super(ResNet50EnhancedAttention, self).__init__()
        
        # Multi-scale residual attention
        self.multi_scale_attention = MultiScaleResidualAttention(in_channels)
        
        # Pyramid pooling attention
        self.pyramid_attention = PyramidPoolingAttention(in_channels)
        
        # Original rock attention (maintain compatibility)
        self.rock_attention = EnhancedRockAttention(in_channels)
        
        # Attention fusion weights
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Final feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels//8),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        # Apply three different attention mechanisms
        ms_out = self.multi_scale_attention(x)
        pyramid_out = self.pyramid_attention(x)
        rock_out = self.rock_attention(x)
        
        # Dynamic weighted fusion
        weights = F.softmax(self.attention_weights, dim=0)
        fused = weights[0] * ms_out + weights[1] * pyramid_out + weights[2] * rock_out
        
        # Final refinement
        refined = self.refinement(fused)
        
        return refined + x  # Residual connection 