import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .attention import EnhancedRockAttention, EfficientNetRockAttention, EfficientNetMinimalAttention, IdentityAttention
from .fpn import FPN, FeatureFusion
from .losses import EnhancedCombinedLoss
from .grad_cam import GradCAM



class ResNet50Model(nn.Module):
    """使用ResNet50作为骨干网络的岩石分类模型"""
    def __init__(self, num_classes, use_attention=True):
        super(ResNet50Model, self).__init__()
        # 使用预训练的ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 特征层通道数
        self.channel_sizes = [256, 512, 1024, 2048]
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.enhanced_rock_attention = EnhancedRockAttention(in_channels=2048)
        
        # 特征金字塔网络
        self.fpn = FPN(
            in_channels_list=self.channel_sizes,
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
        for param in list(self.resnet.parameters())[:-100]:
            param.requires_grad = False
            
        # 损失函数
        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)
        
        # GradCAM可视化 - 使用最后一个卷积层作为目标层
        self.grad_cam = GradCAM(self, self.resnet.layer4)
    
    def _extract_features(self, x):
        """提取ResNet50的各个阶段特征"""
        features = []
        
        # 阶段1：初始卷积和池化
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # 阶段2-5
        x = self.resnet.layer1(x)  # 256通道
        features.append(x)
        
        x = self.resnet.layer2(x)  # 512通道
        features.append(x)
        
        x = self.resnet.layer3(x)  # 1024通道
        features.append(x)
        
        x = self.resnet.layer4(x)  # 2048通道
        if self.use_attention:
            x = self.enhanced_rock_attention(x)
        features.append(x)
        
        return features
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        # 提取特征
        features = self._extract_features(x)
        
        # 特征金字塔增强
        fpn_features = self.fpn(features)
        
        # 特征融合
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # 分类
        logits = self.fc(fused_features)
        
        # 如果是训练模式且提供了标签，计算损失
        if self.training and labels is not None:
            loss = self.criterion(
                fused_features,  # 特征
                logits,          # 分类输出
                labels,          # 标签
                class_accuracies=class_accuracies,
            )
            return logits, loss
        
        return logits, fused_features

class EfficientNetB4Model(nn.Module):
    """使用EfficientNetB4作为骨干网络的岩石分类模型"""
    def __init__(self, num_classes, use_attention=True, attention_type='minimal'):
        super(EfficientNetB4Model, self).__init__()
        # 使用预训练的EfficientNetB4
        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # EfficientNetB4的特征层通道数
        self.channel_sizes = [24, 56, 160, 1792]
        
        # 注意力机制 - 支持多种attention类型
        self.use_attention = use_attention
        self.attention_type = attention_type
        if use_attention:
            if attention_type == 'minimal':
                self.enhanced_rock_attention = EfficientNetMinimalAttention(in_channels=1792)
            elif attention_type == 'identity':
                self.enhanced_rock_attention = IdentityAttention(in_channels=1792)
            elif attention_type == 'rock':
                self.enhanced_rock_attention = EfficientNetRockAttention(in_channels=1792)
            else:
                # 默认使用最保守的版本
                self.enhanced_rock_attention = EfficientNetMinimalAttention(in_channels=1792)
        
        # 特征预处理层（统一空间维度和通道数）
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((35, 35))
            ),
            nn.Sequential(
                nn.Conv2d(56, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((17, 17))
            ),
            nn.Sequential(
                nn.Conv2d(160, 1024, 1),
                nn.BatchNorm2d(1024),
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
        
        # 特征金字塔网络
        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
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
            
        # 损失函数
        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)
        
        # GradCAM可视化 - 使用最后一个特征层作为目标层
        # EfficientNet的特征提取器通常是一个Sequential，获取最后一层
        target_layer = self.efficientnet.features[-1]
        self.grad_cam = GradCAM(self, target_layer)
    
    def _extract_features(self, x):
        """提取EfficientNetB4的中间特征"""
        features = []
        current_feature = x
        
        # 获取特征提取器的所有层
        layers = list(self.efficientnet.features)
        
        # 第一阶段：24通道
        for layer in layers[:2]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[0](current_feature))
        
        # 第二阶段：56通道
        for layer in layers[2:4]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[1](current_feature))
        
        # 第三阶段：160通道
        for layer in layers[4:6]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[2](current_feature))
        
        # 第四阶段：1792通道
        for layer in layers[6:]:
            current_feature = layer(current_feature)
            
        if self.use_attention:
            current_feature = self.enhanced_rock_attention(current_feature)
        features.append(self.preprocess[3](current_feature))
        
        return features
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        # 确保输入尺寸正确 (EfficientNet-B4最佳输入尺寸为380x380)
        if x.size(2) != 380 or x.size(3) != 380:
            x = F.interpolate(x, size=(380, 380), mode='bilinear', align_corners=False)
            
        # 提取特征
        features = self._extract_features(x)
        
        # 特征金字塔增强
        fpn_features = self.fpn(features)
        
        # 特征融合
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # 分类
        logits = self.fc(fused_features)
        
        # 如果是训练模式且提供了标签，计算损失
        if self.training and labels is not None:
            loss = self.criterion(
                fused_features,  # 特征
                logits,          # 分类输出
                labels,          # 标签
                class_accuracies=class_accuracies
            )
            return logits, loss
        
        return logits, fused_features

class InceptionV3Model(nn.Module):
    """使用InceptionV3作为骨干网络的岩石分类模型"""
    def __init__(self, num_classes, use_attention=True):
        super(InceptionV3Model, self).__init__()
        # 使用预训练的Inception V3
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
        
        # 冻结部分层
        for param in list(self.inception.parameters())[:-150]:
            param.requires_grad = False
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.enhanced_rock_attention = EnhancedRockAttention(in_channels=2048)
        
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
        
        # 辅助分类器
        self.aux_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 损失函数
        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)
        
        # GradCAM可视化 - 使用最后一个混合层作为目标层
        self.grad_cam = GradCAM(self, self.inception.Mixed_7c)
    
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
        aux = x if self.training and self.inception.aux_logits else None
        
        # 第四阶段：Mixed_7a (1280 channels)
        x = self.inception.Mixed_7a(x)  # 1280
        features.append(x)  # 1280 channels
        
        # 第五阶段：Mixed_7b/c (2048 channels)
        x = self.inception.Mixed_7b(x)  # 2048
        x = self.inception.Mixed_7c(x)  # 2048
        
        # 应用注意力机制
        if self.use_attention:
            x = self.enhanced_rock_attention(x)
        features.append(x)  # 2048 channels
        
        return features, aux
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        # 确保输入尺寸正确 (InceptionV3需要299x299)
        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
        # 提取特征
        features, aux = self._extract_features(x)
        
        # 特征金字塔增强
        fpn_features = self.fpn(features)
        
        # 特征融合
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # 分类
        logits = self.fc(fused_features)
        
        # 辅助分类器（如果在训练阶段）
        if self.training and aux is not None:
            aux_logits = self.aux_fc(F.adaptive_avg_pool2d(aux, (1, 1)).view(aux.size(0), -1))
        else:
            aux_logits = None
        
        # 如果是训练模式且提供了标签，计算损失
        if self.training and labels is not None:
            # 主分类器损失
            main_loss = self.criterion(
                fused_features,  # 特征
                logits,          # 分类输出
                labels,          # 标签
                class_accuracies=class_accuracies
            )
            
            # 辅助分类器损失（如果有辅助分类器）
            if aux_logits is not None:
                aux_loss = 0.3 * F.cross_entropy(aux_logits, labels)
                loss = main_loss + aux_loss
            else:
                loss = main_loss
                
            return logits, loss
        
        return logits, fused_features

# 无注意力机制的模型版本
def create_model_without_attention(model_name, num_classes):
    """创建不带注意力机制的模型"""
    if model_name == 'resnet50':
        return ResNet50Model(num_classes, use_attention=False)
    elif model_name == 'resnet50_optimized':
        return OptimizedResNet50Model(num_classes, use_attention=False)
    elif model_name == 'efficientnet_b4':
        return EfficientNetB4Model(num_classes, use_attention=False)
    elif model_name == 'inceptionv3':
        return InceptionV3Model(num_classes, use_attention=False)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

# 创建模型函数
def create_model(model_name, num_classes, use_attention=True):
    """创建指定类型的模型"""
    if model_name == 'resnet50':
        return ResNet50Model(num_classes, use_attention)
    elif model_name == 'resnet50_optimized':
        return OptimizedResNet50Model(num_classes, use_attention)
    elif model_name == 'efficientnet_b4':
        return EfficientNetB4Model(num_classes, use_attention)
    elif model_name == 'inceptionv3':
        return InceptionV3Model(num_classes, use_attention)
    elif model_name == 'ensemble':
        # 导入EnsembleModel并返回
        from .ensemble_model import EnsembleModel
        return EnsembleModel(num_classes)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

class OptimizedResNet50Model(nn.Module):
    """优化版ResNet50模型 - 专门针对岩石分类任务优化"""
    def __init__(self, num_classes, use_attention=True):
        super(OptimizedResNet50Model, self).__init__()
        
        # 使用预训练的ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 特征层通道数
        self.channel_sizes = [256, 512, 1024, 2048]
        
        # 注意力机制 - 使用新的优化版注意力
        self.use_attention = use_attention
        if use_attention:
            from .attention import ResNet50EnhancedAttention
            self.enhanced_attention = ResNet50EnhancedAttention(in_channels=2048)
            
            # 在中间层也添加注意力（可选）
            self.mid_attention_1024 = ResNet50EnhancedAttention(in_channels=1024)
            self.mid_attention_512 = ResNet50EnhancedAttention(in_channels=512)
        
        # 改进的特征金字塔网络
        self.fpn = FPN(
            in_channels_list=self.channel_sizes,
            out_channels=512
        )
        
        # 多层次特征融合
        self.multi_level_fusion = nn.Sequential(
            nn.Conv2d(512 * 4, 1024, 1),  # 融合4个层次的特征
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 增强的分类头
        self.classifier = nn.Sequential(
            # 全局平均池化 + 全局最大池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # 第一层
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            
            # 第二层
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            
            # 第三层（辅助特征提取）
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            
            # 输出层
            nn.Linear(256, num_classes)
        )
        
        # 辅助分类头（用于深度监督）
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),  # 1024通道来自layer3
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 冻结策略 - 更加精细的冻结
        self._freeze_layers()
        
        # 使用优化的损失函数
        from .losses import ResNet50OptimizedLoss
        self.criterion = ResNet50OptimizedLoss(num_classes=num_classes, feat_dim=512)
        
        # GradCAM可视化
        self.grad_cam = GradCAM(self, self.resnet.layer4)
    
    def _freeze_layers(self):
        """精细化的层冻结策略"""
        # 冻结早期层，保留后期层的可训练性
        layers_to_freeze = [
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.layer1,
            # layer2前半部分冻结
            *list(self.resnet.layer2.children())[:2],
        ]
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def _extract_features(self, x):
        """提取ResNet50的各个阶段特征"""
        features = []
        
        # 阶段1：初始卷积和池化
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # 阶段2-5，在关键层应用注意力
        x = self.resnet.layer1(x)  # 256通道
        features.append(x)
        
        x = self.resnet.layer2(x)  # 512通道
        if self.use_attention:
            x = self.mid_attention_512(x)
        features.append(x)
        
        x = self.resnet.layer3(x)  # 1024通道
        if self.use_attention:
            x = self.mid_attention_1024(x)
        features.append(x)
        
        x = self.resnet.layer4(x)  # 2048通道
        if self.use_attention:
            x = self.enhanced_attention(x)
        features.append(x)
        
        return features
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        batch_size = x.size(0)
        
        # 提取多层次特征
        features = self._extract_features(x)
        
        # 特征金字塔增强
        fpn_features = self.fpn(features)
        
        # 多层次特征融合
        # 将所有FPN特征调整到相同尺寸并拼接
        target_size = fpn_features[0].shape[2:]
        aligned_features = []
        
        for feat in fpn_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 拼接所有特征
        fused_features = torch.cat(aligned_features, dim=1)
        fused_features = self.multi_level_fusion(fused_features)
        
        # 主分类
        logits = self.classifier(fused_features)
        
        # 辅助分类（使用layer3的特征）
        aux_logits = None
        if self.training:
            aux_logits = self.aux_classifier(features[2])  # layer3特征
        
        # 如果是训练模式且提供了标签，计算损失
        if self.training and labels is not None:
            # 主损失
            main_loss = self.criterion(fused_features, logits, labels, class_accuracies)
            
            # 辅助损失
            aux_loss = 0
            if aux_logits is not None:
                aux_loss = F.cross_entropy(aux_logits, labels) * 0.3
            
            total_loss = main_loss + aux_loss
            return logits, total_loss
        
        return logits, fused_features 