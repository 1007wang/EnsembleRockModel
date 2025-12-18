import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .fpn import FPN, FeatureFusion
from .losses import KnowledgeDistillationLoss, EnhancedCombinedLoss, AdaptiveFocalLoss
from .contrastive_losses import CombinedAdaptiveContrastiveLoss
from .grad_cam import GradCAM  # 导入Grad-CAM类
from .residual_modules import ResidualFeatureFusion  # 🔥 阶段一：引入残差特征融合

class EnhancedInceptionV3(nn.Module):
    """增强版Inception V3"""
    def __init__(self, num_classes):
        super(EnhancedInceptionV3, self).__init__()
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
        
        # 冻结部分层
        for param in list(self.inception.parameters())[:-150]:
            param.requires_grad = False
        
        # 移除外挂注意力模块 - 根据新的技术方案
        
        # 特征金字塔
        self.fpn = FPN(
            in_channels_list=[288, 768, 1280, 2048],
            out_channels=512
        )
        
        # 特征融合
        self.fusion = FeatureFusion(channels=512)
        
        # 分类头
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _extract_features(self, x):
        """提取Inception V3的中间特征"""
        features = []
        
        # 第一阶段：Conv2d layers
        x = self.inception.Conv2d_1a_3x3(x)  # 32
        x = self.inception.Conv2d_2a_3x3(x)  # 32
        x = self.inception.Conv2d_2b_3x3(x)  # 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)  # 80
        x = self.inception.Conv2d_4a_3x3(x)  # 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # 第二阶段：Mixed_5 layers (288 channels)
        x = self.inception.Mixed_5b(x)  # 256
        x = self.inception.Mixed_5c(x)  # 288
        x = self.inception.Mixed_5d(x)  # 288
        features.append(x)  # 288 channels
        
        # 第三阶段：Mixed_6 layers (768 channels)
        x = self.inception.Mixed_6a(x)  # 768
        x = self.inception.Mixed_6b(x)  # 768
        x = self.inception.Mixed_6c(x)  # 768
        x = self.inception.Mixed_6d(x)  # 768
        x = self.inception.Mixed_6e(x)  # 768
        features.append(x)  # 768 channels
        
        # 保存辅助分类器的输入
        aux = None
        if self.training and self.inception.aux_logits:
            aux = self.inception.AuxLogits(x)
        
        # 第四阶段：Mixed_7a (1280 channels)
        x = self.inception.Mixed_7a(x)  # 1280
        features.append(x)  # 1280 channels
        
        # 第五阶段：Mixed_7b/c (2048 channels)
        x = self.inception.Mixed_7b(x)  # 2048
        x = self.inception.Mixed_7c(x)  # 2048
        features.append(x)  # 2048 channels
        
        # 移除注意力机制，保持原始特征
        
        return features, aux
    
    def forward(self, x):
        # 提取特征
        features, aux = self._extract_features(x)
        
        # 特征金字塔增强
        fpn_features = self.fpn(features)
        
        # 特征融合
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # 分类
        x = self.fc(fused_features)
        
        if self.training and aux is not None:
            return x, fused_features, aux
        return x, fused_features

class EfficientNetB4Enhanced(nn.Module):
    """增强版EfficientNet-B4"""
    def __init__(self, num_classes):
        super(EfficientNetB4Enhanced, self).__init__()
        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # EfficientNet-B4的特征层通道数
        self.channel_sizes = [24, 56, 160, 1792]
        
        # 移除外挂注意力模块，保留EfficientNet原生SE模块
        
        # 特征预处理层（统一空间维度和通道数）
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 288, 1),
                nn.BatchNorm2d(288),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((35, 35))
            ),
            nn.Sequential(
                nn.Conv2d(56, 768, 1),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((17, 17))
            ),
            nn.Sequential(
                nn.Conv2d(160, 1280, 1),
                nn.BatchNorm2d(1280),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 8))
            ),
            nn.Sequential(
                nn.Conv2d(1792, 2048, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 8))
            )
        ])
        
        # 特征金字塔
        self.fpn = FPN(
            in_channels_list=[288, 768, 1280, 2048],
            out_channels=512
        )
        
        # 特征融合
        self.fusion = FeatureFusion(channels=512)
        
        # 分类头
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 冻结部分层
        for param in list(self.efficientnet.parameters())[:-100]:
            param.requires_grad = False
    
    def _extract_features(self, x):
        """提取EfficientNet-B4的中间特征"""
        features = []
        current_feature = x
        
        # 获取特征提取器的所有层
        layers = list(self.efficientnet.features)
        
        # 第一阶段：24通道
        for layer in layers[:2]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # 第二阶段：56通道
        for layer in layers[2:4]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # 第三阶段：160通道
        for layer in layers[4:6]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # 第四阶段：1792通道
        for layer in layers[6:]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # 保留EfficientNet原生特征，不添加外挂注意力
        
        # 预处理特征，使其与Inception V3的特征维度匹配
        processed_features = []
        for feat, preprocess in zip(features, self.preprocess):
            processed_features.append(preprocess(feat))
        
        return processed_features
    
    def forward(self, x):
        # 确保输入尺寸正确 (EfficientNet-B4需要380x380)
        if x.shape[-1] != 380:
            x = F.interpolate(x, size=(380, 380), mode='bilinear', align_corners=True)
        
        # 提取并预处理特征
        features = self._extract_features(x)
        
        # 特征金字塔增强
        fpn_features = self.fpn(features)
        
        # 特征融合
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # 分类
        x = self.fc(fused_features)
        
        return x, fused_features

class EnsembleModel(nn.Module):
    """集成模型"""
    def __init__(self, num_classes, temperature=4.0):
        super(EnsembleModel, self).__init__()
        self.inception = EnhancedInceptionV3(num_classes)
        self.efficientnet = EfficientNetB4Enhanced(num_classes)
        
        # 使用增强版组合损失函数
        self.combined_loss = EnhancedCombinedLoss(num_classes, feat_dim=512)
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature)
        self.ce = nn.CrossEntropyLoss()
        
        # 添加对比学习和领域自适应损失
        self.adaptive_contrastive = CombinedAdaptiveContrastiveLoss(
            feature_dim=512,
            num_classes=num_classes,
            temperature=0.07,
            memory_bank_size=4096
        )
        
        # Grad-CAM
        self.grad_cam = None  # Grad-CAM实例将会在forward中创建
        
        # 模型融合权重 - 初始化为相等权重
        self.weight_inception = nn.Parameter(torch.FloatTensor([0.5]))
        self.weight_efficientnet = nn.Parameter(torch.FloatTensor([0.5]))
        
        # 特征对齐层 - 统一到相同尺寸 H×W×C
        self.feature_alignment = nn.ModuleDict({
            'inception': nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),  # 统一空间尺寸到8x8
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            'efficientnet': nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),  # 统一空间尺寸到8x8
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        })
        
        # 分支级门控
        self.branch_gating = BranchGating(feature_dim=512)
        
        # 🔥 阶段一改进：使用残差式特征融合
        self.feature_fusion = ResidualFeatureFusion(feature_dim=512)
        # 原版：self.feature_fusion = AdaptiveFeatureFusion(feature_dim=512, fusion_type='weighted')
        
        # 残差式注意力
        self.residual_attention = ResidualAttentionBlock(feature_dim=512)
        
        # 添加投影头，用于对比学习
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        
        # 存储类别准确率
        self.class_accuracies = None
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None, contrast_weight=0.1):
        # 更新类别准确率
        if class_accuracies is not None:
            self.class_accuracies = class_accuracies
            
        if self.training and labels is not None:
            # 获取Inception输出
            inception_outputs = self.inception(x)
            if len(inception_outputs) == 3:
                inception_logits, inception_features, aux_logits = inception_outputs
            else:
                inception_logits, inception_features = inception_outputs
                aux_logits = None
            
            # 获取EfficientNet输出
            efficientnet_logits, efficientnet_features = self.efficientnet(x)
            
            # 特征对齐到统一尺寸 H×W×C
            aligned_inception = self.feature_alignment['inception'](inception_features)
            aligned_efficientnet = self.feature_alignment['efficientnet'](efficientnet_features)
            
            # 分支级门控
            gated_features = self.branch_gating(aligned_inception, aligned_efficientnet)
            
            # 自适应特征融合
            fused_features = self.feature_fusion(aligned_inception, aligned_efficientnet)
            
            # 残差式注意力
            attention_features = self.residual_attention(fused_features)
            
            # 加权融合
            weights = F.softmax(torch.stack([
                self.weight_inception,
                self.weight_efficientnet
            ]), dim=0)
            
            ensemble_logits = (
                weights[0] * inception_logits +
                weights[1] * efficientnet_logits
            )
            
            # 计算各个模型的损失，使用对齐后的特征
            inception_loss = self.combined_loss(
                aligned_inception, 
                inception_logits, 
                labels, 
                class_accuracies=self.class_accuracies
            )
            
            efficientnet_loss = self.combined_loss(
                aligned_efficientnet, 
                efficientnet_logits, 
                labels,
                class_accuracies=self.class_accuracies
            )
            
            # 使用自适应Focal Loss
            ce_loss = AdaptiveFocalLoss(num_classes=inception_logits.size(1), gamma=1.5, smoothing=0.2)
            if self.class_accuracies is not None:
                ce_loss.update_weights(self.class_accuracies)
            ensemble_loss = ce_loss(ensemble_logits, labels)
            
            # 计算知识蒸馏损失
            kd_loss = self.kd_loss(efficientnet_logits, inception_logits, labels)
            
            # 生成对比学习特征，使用注意力增强后的特征
            inception_feat = F.adaptive_avg_pool2d(aligned_inception, (1, 1)).squeeze(-1).squeeze(-1)
            efficientnet_feat = F.adaptive_avg_pool2d(aligned_efficientnet, (1, 1)).squeeze(-1).squeeze(-1)
            attention_feat = F.adaptive_avg_pool2d(attention_features, (1, 1)).squeeze(-1).squeeze(-1)
            
            # 投影特征
            proj_inception = self.projection(inception_feat)
            proj_efficientnet = self.projection(efficientnet_feat)
            
            # 计算对比学习和领域自适应损失，使用注意力增强特征
            contrast_domain_loss = self.adaptive_contrastive(
                features=self.projection(attention_feat),  # 使用注意力增强后的特征
                labels=labels,
                domain_features=torch.cat([inception_feat, efficientnet_feat, attention_feat], dim=0),
                alpha=alpha
            )
            
            # 总损失，使用动态对比学习权重
            loss = (inception_loss + 
                   efficientnet_loss + 
                   ensemble_loss + 
                   0.5 * kd_loss + 
                   contrast_weight * 1.5 * contrast_domain_loss)  # 使用动态对比学习权重
            
            # 添加辅助损失
            if aux_logits is not None:
                aux_loss = ce_loss(aux_logits, labels)
                loss = loss + 0.4 * aux_loss
            
            return ensemble_logits, loss
        else:
            # 推理模式
            inception_logits, inception_features = self.inception(x)
            efficientnet_logits, efficientnet_features = self.efficientnet(x)
            
            # 特征对齐到统一尺寸 H×W×C
            aligned_inception = self.feature_alignment['inception'](inception_features)
            aligned_efficientnet = self.feature_alignment['efficientnet'](efficientnet_features)
            
            # 分支级门控
            gated_features = self.branch_gating(aligned_inception, aligned_efficientnet)
            
            # 自适应特征融合
            fused_features = self.feature_fusion(aligned_inception, aligned_efficientnet)
            
            # 残差式注意力
            attention_features = self.residual_attention(fused_features)
            
            # 加权融合logits
            weights = F.softmax(torch.stack([
                self.weight_inception,
                self.weight_efficientnet
            ]), dim=0)
            
            ensemble_logits = (
                weights[0] * inception_logits +
                weights[1] * efficientnet_logits
            )
            
            # 创建 Grad-CAM 实例，使用最后一个卷积层作为目标层
            self.grad_cam = GradCAM(self, self.inception.inception.Mixed_7c) 
            
            return ensemble_logits, attention_features  # 返回注意力增强后的特征


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块"""
    def __init__(self, feature_dim=512, fusion_type='weighted'):
        super(AdaptiveFeatureFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # 拼接后的通道数是原来的2倍
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'weighted':
            # 逐通道加权
            self.channel_weights = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, f1, f2):
        if self.fusion_type == 'concat':
            fused = torch.cat([f1, f2], dim=1)
            return self.fusion_conv(fused)
        else:
            combined = torch.cat([f1, f2], dim=1)
            weights = self.channel_weights(combined)
            return f1 * weights + f2 * (1 - weights)


class ResidualAttentionBlock(nn.Module):
    """残差式注意力模块"""
    def __init__(self, feature_dim=512, reduction=16):
        super(ResidualAttentionBlock, self).__init__()
        
        # 通道注意力：全局平均池化 → MLP → Sigmoid
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # 轻量空间注意力：depthwise 3x3 conv + Sigmoid
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # 残差连接权重
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 通道注意力权重
        self.beta = nn.Parameter(torch.tensor(0.3))   # 空间注意力权重
    
    def forward(self, x):
        # 通道注意力：y = x * s_c + x（残差）
        s_c = self.channel_attention(x)
        channel_refined = x * s_c + x  # 残差连接
        
        # 空间注意力：同样残差
        s_s = self.spatial_attention(channel_refined)
        spatial_refined = channel_refined * s_s + channel_refined
        
        # 最终残差融合，防止过度调制
        return self.alpha * spatial_refined + (1 - self.alpha) * x


class BranchGating(nn.Module):
    """分支级门控机制"""
    def __init__(self, feature_dim=512):
        super(BranchGating, self).__init__()
        
        # 分支重要性评估
        self.branch_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 2, 1),  # 输出两个分支的权重
            nn.Softmax(dim=1)
        )
        
        # 分支特征增强
        self.branch_enhancer = nn.ModuleDict({
            'inception': nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ),
            'efficientnet': nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
        })
    
    def forward(self, f_inception, f_efficientnet):
        # 评估分支重要性
        combined = torch.cat([f_inception, f_efficientnet], dim=1)
        branch_weights = self.branch_evaluator(combined)  # [B, 2, 1, 1]
        
        # 增强各分支特征
        enhanced_inception = self.branch_enhancer['inception'](f_inception)
        enhanced_efficientnet = self.branch_enhancer['efficientnet'](f_efficientnet)
        
        # 门控融合
        gated_inception = enhanced_inception * branch_weights[:, 0:1]
        gated_efficientnet = enhanced_efficientnet * branch_weights[:, 1:2]
        
        return gated_inception + gated_efficientnet 