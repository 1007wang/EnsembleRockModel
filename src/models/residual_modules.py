"""
残差增强模块 - 为集成模型引入ResNet残差连接思想
包含4层渐进式残差增强：
1. 残差FPN (ResidualFPN)
2. 残差特征对齐 (ResidualFeatureAlignment)
3. 残差特征融合 (ResidualFeatureFusion)
4. 残差分类头 (ResidualClassificationHead)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """标准残差块 - 类似ResNet的基本块"""
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，使用1x1卷积进行维度匹配
        if use_1x1conv or in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out


class BottleneckResidualBlock(nn.Module):
    """瓶颈残差块 - 类似ResNet50的瓶颈块，参数更少"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleneckResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 维度匹配
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out


class ResidualFPN(nn.Module):
    """残差增强的特征金字塔网络
    
    改进点：
    1. 横向连接使用残差块而非单个卷积
    2. 输出卷积使用残差块
    3. 跨层残差连接（dense connection思想）
    """
    def __init__(self, in_channels_list, out_channels, use_bottleneck=True):
        super(ResidualFPN, self).__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.output_residual_blocks = nn.ModuleList()
        
        # 使用残差块进行横向连接
        for in_channels in in_channels_list:
            if use_bottleneck:
                # 使用瓶颈块，参数更少
                lateral_conv = BottleneckResidualBlock(
                    in_channels, 
                    out_channels // 4, 
                    out_channels
                )
            else:
                # 使用标准残差块
                lateral_conv = ResidualBlock(
                    in_channels, 
                    out_channels, 
                    use_1x1conv=True
                )
            
            self.lateral_convs.append(lateral_conv)
            
            # 输出使用残差块增强
            self.output_residual_blocks.append(
                ResidualBlock(out_channels, out_channels)
            )
        
        # 跨层残差连接的融合模块
        self.cross_layer_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list) - 1)
        ])
    
    def forward(self, features):
        """
        Args:
            features: 特征列表，从浅到深 [C3, C4, C5, C6]
        Returns:
            增强后的多尺度特征列表 [P3, P4, P5, P6]
        """
        # 横向连接（使用残差块）
        laterals = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(lateral_conv(feature))
        
        # 自顶向下路径 + 跨层残差连接
        enhanced_laterals = [laterals[-1]]  # 最顶层
        
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样
            upsampled = F.interpolate(
                enhanced_laterals[-1],
                size=laterals[i - 1].shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            
            # 融合：当前层 + 上采样层 + 跨层残差
            fused = laterals[i - 1] + upsampled
            
            # 如果不是最底层，添加跨层密集连接
            if i > 1:
                # 将最顶层特征也融合进来（跨层连接）
                top_feature = F.interpolate(
                    laterals[-1],
                    size=laterals[i - 1].shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                # 拼接后降维
                cross_layer = self.cross_layer_fusion[i - 1](
                    torch.cat([fused, top_feature], dim=1)
                )
                fused = fused + cross_layer  # 残差连接
            
            enhanced_laterals.append(fused)
        
        enhanced_laterals = enhanced_laterals[::-1]  # 反转回从浅到深
        
        # 使用残差块进一步增强每一层
        results = []
        for lateral, residual_block in zip(enhanced_laterals, self.output_residual_blocks):
            results.append(residual_block(lateral))
        
        return results


class ResidualFeatureAlignment(nn.Module):
    """残差式特征对齐层
    
    改进点：
    1. 使用残差块代替单个卷积
    2. 保留原始特征信息
    3. 自适应空间尺寸调整
    """
    def __init__(self, in_channels, out_channels, target_size=(8, 8)):
        super(ResidualFeatureAlignment, self).__init__()
        
        self.target_size = target_size
        
        # 空间对齐
        self.spatial_align = nn.AdaptiveAvgPool2d(target_size)
        
        # 通道对齐（使用残差块）
        if in_channels != out_channels:
            self.channel_align = nn.Sequential(
                ResidualBlock(in_channels, out_channels, use_1x1conv=True),
                ResidualBlock(out_channels, out_channels)
            )
        else:
            self.channel_align = ResidualBlock(in_channels, out_channels)
        
        # 特征增强
        self.enhancement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # 空间对齐
        x = self.spatial_align(x)
        
        # 通道对齐（带残差）
        aligned = self.channel_align(x)
        
        # 特征增强（带残差连接）
        enhanced = self.enhancement(aligned)
        output = aligned + enhanced  # 残差连接
        
        return F.relu(output)


class ResidualFeatureFusion(nn.Module):
    """残差式特征融合
    
    改进点：
    1. 多路径融合（加法、拼接、门控）
    2. 每个路径都有残差保护
    3. 自适应权重学习
    """
    def __init__(self, feature_dim=512):
        super(ResidualFeatureFusion, self).__init__()
        
        # 路径1：加权加法路径
        self.add_path = nn.Sequential(
            ResidualBlock(feature_dim, feature_dim),
            ResidualBlock(feature_dim, feature_dim)
        )
        
        # 路径2：拼接融合路径
        self.concat_path = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(feature_dim, feature_dim)
        )
        
        # 路径3：门控融合路径
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 多路径自适应加权
        self.path_weights = nn.Parameter(torch.ones(3) / 3)
        
        # 最终融合
        self.final_fusion = ResidualBlock(feature_dim, feature_dim)
    
    def forward(self, f1, f2):
        """
        Args:
            f1, f2: 两个分支的特征 [B, C, H, W]
        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 确保空间尺寸一致
        if f1.shape[-2:] != f2.shape[-2:]:
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        
        # 路径1：加权加法
        add_fused = (f1 + f2) / 2
        add_output = self.add_path(add_fused)
        
        # 路径2：拼接融合
        concat_input = torch.cat([f1, f2], dim=1)
        concat_output = self.concat_path(concat_input)
        
        # 路径3：门控融合
        gate_weights = self.gate_generator(concat_input)  # [B, 2, 1, 1]
        gate_output = f1 * gate_weights[:, 0:1] + f2 * gate_weights[:, 1:2]
        
        # 多路径加权融合
        weights = F.softmax(self.path_weights, dim=0)
        multi_path_fused = (
            weights[0] * add_output + 
            weights[1] * concat_output + 
            weights[2] * gate_output
        )
        
        # 最终残差增强
        final_output = self.final_fusion(multi_path_fused)
        
        # 全局残差连接（保留原始信息）
        return final_output + (f1 + f2) / 2


class ResidualMLPBlock(nn.Module):
    """MLP残差块 - 用于分类头"""
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualMLPBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # 维度匹配
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out


class ResidualClassificationHead(nn.Module):
    """残差式分类头
    
    改进点：
    1. 使用残差MLP块代替线性堆叠
    2. 多尺度特征融合
    3. 深度监督（可选）
    """
    def __init__(self, in_features=512, hidden_dims=[1024, 512], 
                 num_classes=50, dropout=0.3, use_deep_supervision=False):
        super(ResidualClassificationHead, self).__init__()
        
        self.use_deep_supervision = use_deep_supervision
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # 多尺度池化（增加感受野多样性）
        self.multi_scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((s, s)) for s in [1, 2, 4]
        ])
        
        # 多尺度特征融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(in_features * (1 + 4 + 16), in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True)
        )
        
        # 残差MLP层
        self.residual_mlp = nn.ModuleList()
        current_dim = in_features
        
        for hidden_dim in hidden_dims:
            self.residual_mlp.append(
                ResidualMLPBlock(current_dim, hidden_dim, dropout=dropout)
            )
            current_dim = hidden_dim
        
        # 最终分类器
        self.classifier = nn.Linear(current_dim, num_classes)
        
        # 深度监督（辅助分类器）
        if use_deep_supervision:
            self.aux_classifiers = nn.ModuleList([
                nn.Linear(hidden_dim, num_classes) 
                for hidden_dim in hidden_dims
            ])
    
    def forward(self, x, return_aux=False):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            return_aux: 是否返回辅助输出（用于深度监督）
        Returns:
            主分类logits, [辅助logits列表]
        """
        # 多尺度池化
        multi_scale_feats = []
        for pool in self.multi_scale_pools:
            pooled = pool(x)
            multi_scale_feats.append(pooled.flatten(1))
        
        # 融合多尺度特征
        fused_feat = torch.cat(multi_scale_feats, dim=1)
        fused_feat = self.scale_fusion(fused_feat)
        
        # 残差MLP处理
        aux_outputs = []
        current_feat = fused_feat
        
        for i, residual_block in enumerate(self.residual_mlp):
            current_feat = residual_block(current_feat)
            
            # 深度监督
            if self.use_deep_supervision and return_aux and self.training:
                aux_outputs.append(self.aux_classifiers[i](current_feat))
        
        # 最终分类
        main_output = self.classifier(current_feat)
        
        if return_aux and self.training and self.use_deep_supervision:
            return main_output, aux_outputs
        
        return main_output


class ResidualBranchGating(nn.Module):
    """残差式分支门控
    
    改进原有的BranchGating，增加残差连接保护
    """
    def __init__(self, feature_dim=512):
        super(ResidualBranchGating, self).__init__()
        
        # 分支重要性评估（使用残差块）
        self.branch_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            ResidualMLPBlock(feature_dim, feature_dim, dropout=0.1),
            nn.Conv2d(feature_dim, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 分支特征增强（使用残差块）
        self.branch_enhancer = nn.ModuleDict({
            'inception': ResidualBlock(feature_dim, feature_dim),
            'efficientnet': ResidualBlock(feature_dim, feature_dim)
        })
        
        # 融合后的残差增强
        self.fusion_enhancer = ResidualBlock(feature_dim, feature_dim)
    
    def forward(self, f_inception, f_efficientnet):
        # 评估分支重要性
        combined = torch.cat([f_inception, f_efficientnet], dim=1)
        branch_weights = self.branch_evaluator(combined)  # [B, 2, 1, 1]
        
        # 增强各分支特征（带残差）
        enhanced_inception = self.branch_enhancer['inception'](f_inception)
        enhanced_efficientnet = self.branch_enhancer['efficientnet'](f_efficientnet)
        
        # 门控融合
        gated = (enhanced_inception * branch_weights[:, 0:1] + 
                 enhanced_efficientnet * branch_weights[:, 1:2])
        
        # 融合后残差增强
        final_output = self.fusion_enhancer(gated)
        
        # 全局残差连接（保留原始信息）
        return final_output + (f_inception + f_efficientnet) / 2


# ============ 工具函数 ============

def replace_module_with_residual(model, module_name, new_module):
    """
    辅助函数：替换模型中的模块为残差版本
    
    Args:
        model: 模型实例
        module_name: 要替换的模块名称（如 'fpn', 'feature_alignment'）
        new_module: 新的残差模块实例
    """
    parts = module_name.split('.')
    parent = model
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    setattr(parent, parts[-1], new_module)
    print(f"✓ 已将 {module_name} 替换为残差增强版本")


def count_parameters(module):
    """统计模块参数量"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("测试残差增强模块")
    print("=" * 50)
    
    # 测试ResidualFPN
    print("\n1. 测试ResidualFPN")
    fpn = ResidualFPN([288, 768, 1280, 2048], 512)
    test_features = [
        torch.randn(2, 288, 35, 35),
        torch.randn(2, 768, 17, 17),
        torch.randn(2, 1280, 8, 8),
        torch.randn(2, 2048, 8, 8)
    ]
    fpn_outputs = fpn(test_features)
    print(f"   输入: {[f.shape for f in test_features]}")
    print(f"   输出: {[f.shape for f in fpn_outputs]}")
    print(f"   参数量: {count_parameters(fpn):,}")
    
    # 测试ResidualFeatureAlignment
    print("\n2. 测试ResidualFeatureAlignment")
    alignment = ResidualFeatureAlignment(512, 512, target_size=(8, 8))
    test_input = torch.randn(2, 512, 17, 17)
    aligned = alignment(test_input)
    print(f"   输入: {test_input.shape}")
    print(f"   输出: {aligned.shape}")
    print(f"   参数量: {count_parameters(alignment):,}")
    
    # 测试ResidualFeatureFusion
    print("\n3. 测试ResidualFeatureFusion")
    fusion = ResidualFeatureFusion(512)
    f1 = torch.randn(2, 512, 8, 8)
    f2 = torch.randn(2, 512, 8, 8)
    fused = fusion(f1, f2)
    print(f"   输入: {f1.shape}, {f2.shape}")
    print(f"   输出: {fused.shape}")
    print(f"   参数量: {count_parameters(fusion):,}")
    
    # 测试ResidualClassificationHead
    print("\n4. 测试ResidualClassificationHead")
    head = ResidualClassificationHead(512, [1024, 512], 50, use_deep_supervision=True)
    test_feat = torch.randn(2, 512, 8, 8)
    head.train()
    main_out, aux_outs = head(test_feat, return_aux=True)
    print(f"   输入: {test_feat.shape}")
    print(f"   主输出: {main_out.shape}")
    print(f"   辅助输出: {[aux.shape for aux in aux_outs]}")
    print(f"   参数量: {count_parameters(head):,}")
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)

