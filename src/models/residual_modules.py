"""
Residual enhancement modules - Introduce ResNet residual connection ideas for ensemble models
Contains 4 progressive residual enhancements:
1. Residual FPN (ResidualFPN)
2. Residual feature alignment (ResidualFeatureAlignment)
3. Residual feature fusion (ResidualFeatureFusion)
4. Residual classification head (ResidualClassificationHead)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block - similar to ResNet's basic block"""
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If input and output channels differ, use 1x1 convolution for dimension matching
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
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class BottleneckResidualBlock(nn.Module):
    """Bottleneck residual block - similar to ResNet50's bottleneck block, fewer parameters"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleneckResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Dimension matching
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
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class ResidualFPN(nn.Module):
    """Residual-enhanced Feature Pyramid Network
    
    Improvements:
    1. Lateral connections use residual blocks instead of single convolutions
    2. Output convolutions use residual blocks
    3. Cross-layer residual connections (dense connection idea)
    """
    def __init__(self, in_channels_list, out_channels, use_bottleneck=True):
        super(ResidualFPN, self).__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.output_residual_blocks = nn.ModuleList()
        
        # Use residual blocks for lateral connections
        for in_channels in in_channels_list:
            if use_bottleneck:
                # Use bottleneck blocks, fewer parameters
                lateral_conv = BottleneckResidualBlock(
                    in_channels, 
                    out_channels // 4, 
                    out_channels
                )
            else:
                # Use standard residual blocks
                lateral_conv = ResidualBlock(
                    in_channels, 
                    out_channels, 
                    use_1x1conv=True
                )
            
            self.lateral_convs.append(lateral_conv)
            
            # Output uses residual blocks for enhancement
            self.output_residual_blocks.append(
                ResidualBlock(out_channels, out_channels)
            )
        
        # Fusion module for cross-layer residual connections
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
            features: Feature list from shallow to deep [C3, C4, C5, C6]
        Returns:
            Enhanced multi-scale feature list [P3, P4, P5, P6]
        """
        # Lateral connections (using residual blocks)
        laterals = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(lateral_conv(feature))
        
        # Top-down path + cross-layer residual connections
        enhanced_laterals = [laterals[-1]]  # Topmost layer
        
        for i in range(len(laterals) - 1, 0, -1):
            # Upsampling
            upsampled = F.interpolate(
                enhanced_laterals[-1],
                size=laterals[i - 1].shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            
            # Fusion: current layer + upsampled layer + cross-layer residual
            fused = laterals[i - 1] + upsampled
            
            # If not the bottommost layer, add cross-layer dense connections
            if i > 1:
                # Also fuse topmost layer features (cross-layer connection)
                top_feature = F.interpolate(
                    laterals[-1],
                    size=laterals[i - 1].shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                # Concatenate then reduce dimensions
                cross_layer = self.cross_layer_fusion[i - 1](
                    torch.cat([fused, top_feature], dim=1)
                )
                fused = fused + cross_layer  # Residual connection
            
            enhanced_laterals.append(fused)
        
        enhanced_laterals = enhanced_laterals[::-1]  # Reverse back to shallow to deep
        
        # Further enhance each layer using residual blocks
        results = []
        for lateral, residual_block in zip(enhanced_laterals, self.output_residual_blocks):
            results.append(residual_block(lateral))
        
        return results


class ResidualFeatureAlignment(nn.Module):
    """Residual feature alignment layer
    
    Improvements:
    1. Use residual blocks instead of single convolutions
    2. Preserve original feature information
    3. Adaptive spatial size adjustment
    """
    def __init__(self, in_channels, out_channels, target_size=(8, 8)):
        super(ResidualFeatureAlignment, self).__init__()
        
        self.target_size = target_size
        
        # Spatial alignment
        self.spatial_align = nn.AdaptiveAvgPool2d(target_size)
        
        # Channel alignment (using residual blocks)
        if in_channels != out_channels:
            self.channel_align = nn.Sequential(
                ResidualBlock(in_channels, out_channels, use_1x1conv=True),
                ResidualBlock(out_channels, out_channels)
            )
        else:
            self.channel_align = ResidualBlock(in_channels, out_channels)
        
        # Feature enhancement
        self.enhancement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # Spatial alignment
        x = self.spatial_align(x)
        
        # Channel alignment (with residual)
        aligned = self.channel_align(x)
        
        # Feature enhancement (with residual connection)
        enhanced = self.enhancement(aligned)
        output = aligned + enhanced  # Residual connection
        
        return F.relu(output)


class ResidualFeatureFusion(nn.Module):
    """Residual feature fusion
    
    Improvements:
    1. Multi-path fusion (addition, concatenation, gating)
    2. Each path has residual protection
    3. Adaptive weight learning
    """
    def __init__(self, feature_dim=512):
        super(ResidualFeatureFusion, self).__init__()
        
        # Path 1: Weighted addition path
        self.add_path = nn.Sequential(
            ResidualBlock(feature_dim, feature_dim),
            ResidualBlock(feature_dim, feature_dim)
        )
        
        # Path 2: Concatenation fusion path
        self.concat_path = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(feature_dim, feature_dim)
        )
        
        # Path 3: Gated fusion path
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Multi-path adaptive weighting
        self.path_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Final fusion
        self.final_fusion = ResidualBlock(feature_dim, feature_dim)
    
    def forward(self, f1, f2):
        """
        Args:
            f1, f2: Features from two branches [B, C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        # Ensure spatial dimensions are consistent
        if f1.shape[-2:] != f2.shape[-2:]:
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        
        # Path 1: Weighted addition
        add_fused = (f1 + f2) / 2
        add_output = self.add_path(add_fused)
        
        # Path 2: Concatenation fusion
        concat_input = torch.cat([f1, f2], dim=1)
        concat_output = self.concat_path(concat_input)
        
        # Path 3: Gated fusion
        gate_weights = self.gate_generator(concat_input)  # [B, 2, 1, 1]
        gate_output = f1 * gate_weights[:, 0:1] + f2 * gate_weights[:, 1:2]
        
        # Multi-path weighted fusion
        weights = F.softmax(self.path_weights, dim=0)
        multi_path_fused = (
            weights[0] * add_output + 
            weights[1] * concat_output + 
            weights[2] * gate_output
        )
        
        # Final residual enhancement
        final_output = self.final_fusion(multi_path_fused)
        
        # Global residual connection (preserve original information)
        return final_output + (f1 + f2) / 2


class ResidualMLPBlock(nn.Module):
    """MLP residual block - for classification head"""
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualMLPBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Dimension matching
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
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class ResidualClassificationHead(nn.Module):
    """Residual classification head
    
    Improvements:
    1. Use residual MLP blocks instead of linear stacking
    2. Multi-scale feature fusion
    3. Deep supervision (optional)
    """
    def __init__(self, in_features=512, hidden_dims=[1024, 512], 
                 num_classes=50, dropout=0.3, use_deep_supervision=False):
        super(ResidualClassificationHead, self).__init__()
        
        self.use_deep_supervision = use_deep_supervision
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Multi-scale pooling (increase receptive field diversity)
        self.multi_scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((s, s)) for s in [1, 2, 4]
        ])
        
        # Multi-scale feature fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(in_features * (1 + 4 + 16), in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True)
        )
        
        # Residual MLP layers
        self.residual_mlp = nn.ModuleList()
        current_dim = in_features
        
        for hidden_dim in hidden_dims:
            self.residual_mlp.append(
                ResidualMLPBlock(current_dim, hidden_dim, dropout=dropout)
            )
            current_dim = hidden_dim
        
        # Final classifier
        self.classifier = nn.Linear(current_dim, num_classes)
        
        # Deep supervision (auxiliary classifiers)
        if use_deep_supervision:
            self.aux_classifiers = nn.ModuleList([
                nn.Linear(hidden_dim, num_classes) 
                for hidden_dim in hidden_dims
            ])
    
    def forward(self, x, return_aux=False):
        """
        Args:
            x: Input features [B, C, H, W]
            return_aux: Whether to return auxiliary outputs (for deep supervision)
        Returns:
            Main classification logits, [auxiliary logits list]
        """
        # Multi-scale pooling
        multi_scale_feats = []
        for pool in self.multi_scale_pools:
            pooled = pool(x)
            multi_scale_feats.append(pooled.flatten(1))
        
        # Fuse multi-scale features
        fused_feat = torch.cat(multi_scale_feats, dim=1)
        fused_feat = self.scale_fusion(fused_feat)
        
        # Residual MLP processing
        aux_outputs = []
        current_feat = fused_feat
        
        for i, residual_block in enumerate(self.residual_mlp):
            current_feat = residual_block(current_feat)
            
            # Deep supervision
            if self.use_deep_supervision and return_aux and self.training:
                aux_outputs.append(self.aux_classifiers[i](current_feat))
        
        # Final classification
        main_output = self.classifier(current_feat)
        
        if return_aux and self.training and self.use_deep_supervision:
            return main_output, aux_outputs
        
        return main_output


class ResidualBranchGating(nn.Module):
    """Residual branch gating
    
    Improve original BranchGating by adding residual connection protection
    """
    def __init__(self, feature_dim=512):
        super(ResidualBranchGating, self).__init__()
        
        # Branch importance evaluation (using residual blocks)
        self.branch_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            ResidualMLPBlock(feature_dim, feature_dim, dropout=0.1),
            nn.Conv2d(feature_dim, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Branch feature enhancement (using residual blocks)
        self.branch_enhancer = nn.ModuleDict({
            'inception': ResidualBlock(feature_dim, feature_dim),
            'efficientnet': ResidualBlock(feature_dim, feature_dim)
        })
        
        # Residual enhancement after fusion
        self.fusion_enhancer = ResidualBlock(feature_dim, feature_dim)
    
    def forward(self, f_inception, f_efficientnet):
        # Evaluate branch importance
        combined = torch.cat([f_inception, f_efficientnet], dim=1)
        branch_weights = self.branch_evaluator(combined)  # [B, 2, 1, 1]
        
        # Enhance each branch feature (with residual)
        enhanced_inception = self.branch_enhancer['inception'](f_inception)
        enhanced_efficientnet = self.branch_enhancer['efficientnet'](f_efficientnet)
        
        # Gated fusion
        gated = (enhanced_inception * branch_weights[:, 0:1] + 
                 enhanced_efficientnet * branch_weights[:, 1:2])
        
        # Residual enhancement after fusion
        final_output = self.fusion_enhancer(gated)
        
        # Global residual connection (preserve original information)
        return final_output + (f_inception + f_efficientnet) / 2


# ============ Utility Functions ============

def replace_module_with_residual(model, module_name, new_module):
    """
    Helper function: Replace module in model with residual version
    
    Args:
        model: Model instance
        module_name: Name of module to replace (e.g., 'fpn', 'feature_alignment')
        new_module: New residual module instance
    """
    parts = module_name.split('.')
    parent = model
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    setattr(parent, parts[-1], new_module)
    print(f"✓ Replaced {module_name} with residual-enhanced version")


def count_parameters(module):
    """Count module parameters"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test code
    print("=" * 50)
    print("Testing residual enhancement modules")
    print("=" * 50)
    
    # Test ResidualFPN
    print("\n1. Testing ResidualFPN")
    fpn = ResidualFPN([288, 768, 1280, 2048], 512)
    test_features = [
        torch.randn(2, 288, 35, 35),
        torch.randn(2, 768, 17, 17),
        torch.randn(2, 1280, 8, 8),
        torch.randn(2, 2048, 8, 8)
    ]
    fpn_outputs = fpn(test_features)
    print(f"   Input: {[f.shape for f in test_features]}")
    print(f"   Output: {[f.shape for f in fpn_outputs]}")
    print(f"   Parameters: {count_parameters(fpn):,}")
    
    # Test ResidualFeatureAlignment
    print("\n2. Testing ResidualFeatureAlignment")
    alignment = ResidualFeatureAlignment(512, 512, target_size=(8, 8))
    test_input = torch.randn(2, 512, 17, 17)
    aligned = alignment(test_input)
    print(f"   Input: {test_input.shape}")
    print(f"   Output: {aligned.shape}")
    print(f"   Parameters: {count_parameters(alignment):,}")
    
    # Test ResidualFeatureFusion
    print("\n3. Testing ResidualFeatureFusion")
    fusion = ResidualFeatureFusion(512)
    f1 = torch.randn(2, 512, 8, 8)
    f2 = torch.randn(2, 512, 8, 8)
    fused = fusion(f1, f2)
    print(f"   Input: {f1.shape}, {f2.shape}")
    print(f"   Output: {fused.shape}")
    print(f"   Parameters: {count_parameters(fusion):,}")
    
    # Test ResidualClassificationHead
    print("\n4. Testing ResidualClassificationHead")
    head = ResidualClassificationHead(512, [1024, 512], 50, use_deep_supervision=True)
    test_feat = torch.randn(2, 512, 8, 8)
    head.train()
    main_out, aux_outs = head(test_feat, return_aux=True)
    print(f"   Input: {test_feat.shape}")
    print(f"   Main output: {main_out.shape}")
    print(f"   Auxiliary outputs: {[aux.shape for aux in aux_outs]}")
    print(f"   Parameters: {count_parameters(head):,}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

