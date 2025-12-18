import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterWeightedSpatialAttention(nn.Module):
    """中心加权的空间注意力模块"""
    def __init__(self, kernel_size=7, sigma=1.0, center_weight_power=0.5):
        super(CenterWeightedSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.sigma = sigma
        self.center_weight_power = center_weight_power  # 控制中心权重的强度
        self.register_buffer('center_weight', None)
        
    def create_center_weight(self, size):
        """创建中心权重矩阵"""
        if self.center_weight is not None and self.center_weight.size() == (1, 1, size[2], size[3]):
            return self.center_weight
            
        center_x = size[2] // 2
        center_y = size[3] // 2
        
        # 确保在正确的设备上创建张量
        x = torch.arange(size[2], device=self.conv.weight.device).float()
        y = torch.arange(size[3], device=self.conv.weight.device).float()
        
        # 使用indexing='ij'参数
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        
        # 计算到中心的距离
        distance = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_distance = torch.sqrt(torch.tensor(center_x**2 + center_y**2).float())
        
        # 生成高斯权重
        weight = torch.exp(-distance**2 / (2 * self.sigma**2))
        
        # 保存到缓冲区
        self.register_buffer('center_weight', weight.unsqueeze(0).unsqueeze(0))
        return self.center_weight

    def forward(self, x):
        # 生成中心权重
        center_weight = self.create_center_weight(x.size())
        
        # 增强中心权重的影响 - 使用指数调整
        enhanced_center_weight = center_weight ** self.center_weight_power
        
        # 计算通道注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 应用卷积和sigmoid
        attention = self.sigmoid(self.conv(x_cat))
        
        # 结合中心权重，增强中心区域的影响
        attention = attention * enhanced_center_weight
        
        return x * attention.expand_as(x)

class EdgeSuppressionModule(nn.Module):
    """边缘抑制模块"""
    def __init__(self, channels, reduction=16):
        super(EdgeSuppressionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.center_attention = CenterWeightedSpatialAttention()
        
    def forward(self, x):
        # 提取边缘特征
        edge_features = self.edge_conv(x)
        
        # 应用中心注意力
        center_weighted = self.center_attention(x)
        
        # 温和的边缘抑制（减少抑制强度）
        edge_weight = torch.sigmoid(edge_features) * 0.2  # 使用sigmoid并降低权重
        return center_weighted * (1 - edge_weight) 


class AdaptiveTextureAttention(nn.Module):
    """自适应纹理注意力模块"""
    def __init__(self, in_channels):
        super(AdaptiveTextureAttention, self).__init__()
        
        # 多尺度纹理特征提取
        self.texture_scales = [3, 5, 7, 9, 11]  # 增加更细粒度的感受野
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
        
        # 自适应权重生成
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(self.texture_scales), 1),
            nn.Softmax(dim=1)
        )
        
        # 修正：确保融合层输入通道数正确
        # 5个尺度，每个尺度输出in_channels//4通道，总共5*in_channels//4通道
        self.fusion = nn.Sequential(
            nn.Conv2d(5*in_channels//4, in_channels, 1),  # 修正通道数
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 提取多尺度纹理特征
        texture_feats = [branch(x) for branch in self.texture_branches]
        
        # 生成自适应权重
        weights = self.weight_generator(x)
        
        # 加权融合纹理特征
        weighted_feats = []
        for i, feat in enumerate(texture_feats):
            weighted_feats.append(feat * weights[:, i:i+1])
        
        fused_texture = torch.cat(weighted_feats, dim=1)
        return self.fusion(fused_texture)

class DynamicSpatialAttention(nn.Module):
    """动态空间注意力模块"""
    def __init__(self, kernel_sizes=[3, 7, 11, 15]):
        super(DynamicSpatialAttention, self).__init__()
        
        self.spatial_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 8, k, padding=k//2),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        
        # 动态注意力图生成
        self.attention_generator = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * 8, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 局部-全局特征融合
        self.local_global_fusion = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取统计特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # 多尺度空间特征
        multi_scale_feats = []
        for branch in self.spatial_branches:
            feat = branch(spatial_features)
            multi_scale_feats.append(feat)
        
        # 融合多尺度特征
        fused_spatial = torch.cat(multi_scale_feats, dim=1)
        attention = self.attention_generator(fused_spatial)
        
        # 局部-全局特征融合
        local_global = self.local_global_fusion(spatial_features)
        
        # 组合注意力
        final_attention = attention * local_global
        
        return x * final_attention

class EnhancedRockAttention(nn.Module):
    """增强版岩石注意力模块"""
    def __init__(self, in_channels):
        super(EnhancedRockAttention, self).__init__()
        
        # 纹理注意力
        self.texture_attention = AdaptiveTextureAttention(in_channels)
        
        # 动态空间注意力
        self.spatial_attention = DynamicSpatialAttention()
        
        # 添加边缘抑制模块
        self.edge_suppression = EdgeSuppressionModule(in_channels)
        
        # 特征重校准
        self.recalibration = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 纹理增强
        texture_enhanced = self.texture_attention(x)
        
        # 空间注意力
        spatial_enhanced = self.spatial_attention(texture_enhanced)
        
        # 温和的边缘抑制模块
        edge_suppressed = self.edge_suppression(spatial_enhanced)
        
        # 特征重校准
        recalibrated = self.recalibration(edge_suppressed)
        
        # 添加残差连接，防止过度抑制
        attention_weight = 0.3  # 降低attention的影响强度
        return x * (1 - attention_weight) + (x * recalibrated) * attention_weight 

class EfficientNetLightAttention(nn.Module):
    """专门为EfficientNet设计的轻量化注意力模块"""
    def __init__(self, in_channels, reduction=8):
        super(EfficientNetLightAttention, self).__init__()
        
        # 轻量化通道注意力（类似SE模块但更温和）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 简化的空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 残差权重控制
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 较小的初始权重
        
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x_sa = x_ca * sa
        
        # 温和的残差连接，避免过度抑制
        return (1 - self.alpha) * x + self.alpha * x_sa

class EfficientNetMinimalAttention(nn.Module):
    """EfficientNet的极简attention - 解决过度抑制问题"""
    def __init__(self, in_channels):
        super(EfficientNetMinimalAttention, self).__init__()
        
        # 只保留最基本的SE模块，没有额外抑制
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 可学习的增强权重，初始化为接近1，以便保持特征
        self.enhancement_weight = nn.Parameter(torch.tensor(1.1))  # 略大于1，轻微增强
        
    def forward(self, x):
        # SE注意力权重
        se_weights = self.se(x)
        
        # 轻微增强而不是抑制
        enhanced = x * se_weights * self.enhancement_weight
        
        # 直接返回增强后的特征，不做额外抑制
        return enhanced

class EfficientNetRockAttention(nn.Module):
    """专门为EfficientNet岩石分类设计的attention模块 - 修复版"""
    def __init__(self, in_channels):
        super(EfficientNetRockAttention, self).__init__()
        
        # 使用极简attention替代复杂的多层抑制
        self.minimal_attention = EfficientNetMinimalAttention(in_channels)
        
    def forward(self, x):
        # 直接使用极简attention，避免多重抑制
        return self.minimal_attention(x)

class IdentityAttention(nn.Module):
    """身份注意力 - 用于测试，基本不改变输入"""
    def __init__(self, in_channels):
        super(IdentityAttention, self).__init__()
        # 只做很小的调整，接近恒等映射
        self.adjustment = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        return x * self.adjustment

class MultiScaleResidualAttention(nn.Module):
    """专门为ResNet50设计的多尺度残差注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleResidualAttention, self).__init__()
        
        # 多尺度特征提取分支
        self.scales = [1, 3, 5, 7]  # 不同尺度的卷积核
        self.multi_scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=scale, 
                         padding=scale//2, groups=in_channels//4),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ) for scale in self.scales
        ])
        
        # 通道注意力 - 使用SE模块的改进版
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 针对岩石纹理优化
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接的权重参数
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        residual = x
        
        # 多尺度特征提取
        multi_scale_feats = []
        for branch in self.multi_scale_branches:
            feat = branch(x)
            multi_scale_feats.append(feat)
        
        # 融合多尺度特征
        fused_multi_scale = torch.cat(multi_scale_feats, dim=1)
        
        # 通道注意力
        channel_att = self.channel_attention(x)
        channel_refined = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(channel_refined, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_refined, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        spatial_refined = channel_refined * spatial_att
        
        # 特征融合
        output = self.fusion(spatial_refined)
        
        # 学习残差连接的权重
        return self.alpha * output + (1 - self.alpha) * residual

class PyramidPoolingAttention(nn.Module):
    """金字塔池化注意力模块 - 专门处理不同尺度的岩石纹理"""
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
        
        # 特征聚合
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
        
        # 融合所有尺度特征
        fused = torch.cat(pyramid_feats, dim=1)
        attention_map = self.aggregation(fused)
        
        return x * attention_map

class ResNet50EnhancedAttention(nn.Module):
    """专门为ResNet50设计的增强注意力模块组合"""
    def __init__(self, in_channels):
        super(ResNet50EnhancedAttention, self).__init__()
        
        # 多尺度残差注意力
        self.multi_scale_attention = MultiScaleResidualAttention(in_channels)
        
        # 金字塔池化注意力
        self.pyramid_attention = PyramidPoolingAttention(in_channels)
        
        # 原有的岩石注意力（保持兼容性）
        self.rock_attention = EnhancedRockAttention(in_channels)
        
        # 注意力融合权重
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        
        # 最终特征细化
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels//8),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        # 应用三种不同的注意力机制
        ms_out = self.multi_scale_attention(x)
        pyramid_out = self.pyramid_attention(x)
        rock_out = self.rock_attention(x)
        
        # 动态加权融合
        weights = F.softmax(self.attention_weights, dim=0)
        fused = weights[0] * ms_out + weights[1] * pyramid_out + weights[2] * rock_out
        
        # 最终细化
        refined = self.refinement(fused)
        
        return refined + x  # 残差连接 