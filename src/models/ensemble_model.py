import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .fpn import FPN, FeatureFusion
from .losses import KnowledgeDistillationLoss, EnhancedCombinedLoss, AdaptiveFocalLoss
from .contrastive_losses import CombinedAdaptiveContrastiveLoss
from .grad_cam import GradCAM  # Import Grad-CAM class
from .residual_modules import ResidualFeatureFusion  # 🔥 Stage 1: Introduce residual feature fusion

class EnhancedInceptionV3(nn.Module):
    """Enhanced Inception V3"""
    def __init__(self, num_classes):
        super(EnhancedInceptionV3, self).__init__()
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
        
        # Freeze some layers
        for param in list(self.inception.parameters())[:-150]:
            param.requires_grad = False
        
        # Remove external attention modules - according to new technical approach
        
        # Feature pyramid
        self.fpn = FPN(
            in_channels_list=[288, 768, 1280, 2048],
            out_channels=512
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(channels=512)
        
        # Classification head
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
        """Extract intermediate features from Inception V3"""
        features = []
        
        # Stage 1: Conv2d layers
        x = self.inception.Conv2d_1a_3x3(x)  # 32
        x = self.inception.Conv2d_2a_3x3(x)  # 32
        x = self.inception.Conv2d_2b_3x3(x)  # 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)  # 80
        x = self.inception.Conv2d_4a_3x3(x)  # 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # Stage 2: Mixed_5 layers (288 channels)
        x = self.inception.Mixed_5b(x)  # 256
        x = self.inception.Mixed_5c(x)  # 288
        x = self.inception.Mixed_5d(x)  # 288
        features.append(x)  # 288 channels
        
        # Stage 3: Mixed_6 layers (768 channels)
        x = self.inception.Mixed_6a(x)  # 768
        x = self.inception.Mixed_6b(x)  # 768
        x = self.inception.Mixed_6c(x)  # 768
        x = self.inception.Mixed_6d(x)  # 768
        x = self.inception.Mixed_6e(x)  # 768
        features.append(x)  # 768 channels
        
        # Save input for auxiliary classifier
        aux = None
        if self.training and self.inception.aux_logits:
            aux = self.inception.AuxLogits(x)
        
        # Stage 4: Mixed_7a (1280 channels)
        x = self.inception.Mixed_7a(x)  # 1280
        features.append(x)  # 1280 channels
        
        # Stage 5: Mixed_7b/c (2048 channels)
        x = self.inception.Mixed_7b(x)  # 2048
        x = self.inception.Mixed_7c(x)  # 2048
        features.append(x)  # 2048 channels
        
        # Remove attention mechanism, preserve original features
        
        return features, aux
    
    def forward(self, x):
        # Extract features
        features, aux = self._extract_features(x)
        
        # Feature pyramid enhancement
        fpn_features = self.fpn(features)
        
        # Feature fusion
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # Classification
        x = self.fc(fused_features)
        
        if self.training and aux is not None:
            return x, fused_features, aux
        return x, fused_features

class EfficientNetB4Enhanced(nn.Module):
    """Enhanced EfficientNet-B4"""
    def __init__(self, num_classes):
        super(EfficientNetB4Enhanced, self).__init__()
        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # EfficientNet-B4 feature layer channel sizes
        self.channel_sizes = [24, 56, 160, 1792]
        
        # Remove external attention modules, preserve EfficientNet native SE modules
        
        # Feature preprocessing layers (unify spatial dimensions and channel counts)
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
        
        # Feature pyramid
        self.fpn = FPN(
            in_channels_list=[288, 768, 1280, 2048],
            out_channels=512
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(channels=512)
        
        # Classification head
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
        
        # Freeze some layers
        for param in list(self.efficientnet.parameters())[:-100]:
            param.requires_grad = False
    
    def _extract_features(self, x):
        """Extract intermediate features from EfficientNet-B4"""
        features = []
        current_feature = x
        
        # Get all layers from feature extractor
        layers = list(self.efficientnet.features)
        
        # Stage 1: 24 channels
        for layer in layers[:2]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # Stage 2: 56 channels
        for layer in layers[2:4]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # Stage 3: 160 channels
        for layer in layers[4:6]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # Stage 4: 1792 channels
        for layer in layers[6:]:
            current_feature = layer(current_feature)
        features.append(current_feature)
        
        # Preserve EfficientNet native features, do not add external attention
        
        # Preprocess features to match Inception V3 feature dimensions
        processed_features = []
        for feat, preprocess in zip(features, self.preprocess):
            processed_features.append(preprocess(feat))
        
        return processed_features
    
    def forward(self, x):
        # Ensure input size is correct (EfficientNet-B4 requires 380x380)
        if x.shape[-1] != 380:
            x = F.interpolate(x, size=(380, 380), mode='bilinear', align_corners=True)
        
        # Extract and preprocess features
        features = self._extract_features(x)
        
        # Feature pyramid enhancement
        fpn_features = self.fpn(features)
        
        # Feature fusion
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # Classification
        x = self.fc(fused_features)
        
        return x, fused_features

class EnsembleModel(nn.Module):
    """Ensemble model"""
    def __init__(self, num_classes, temperature=4.0):
        super(EnsembleModel, self).__init__()
        self.inception = EnhancedInceptionV3(num_classes)
        self.efficientnet = EfficientNetB4Enhanced(num_classes)
        
        # Use enhanced combined loss function
        self.combined_loss = EnhancedCombinedLoss(num_classes, feat_dim=512)
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature)
        self.ce = nn.CrossEntropyLoss()
        
        # Add contrastive learning and domain adaptation loss
        self.adaptive_contrastive = CombinedAdaptiveContrastiveLoss(
            feature_dim=512,
            num_classes=num_classes,
            temperature=0.07,
            memory_bank_size=4096
        )
        
        # Grad-CAM
        self.grad_cam = None  # Grad-CAM instance will be created in forward
        
        # Model fusion weights - initialized to equal weights
        self.weight_inception = nn.Parameter(torch.FloatTensor([0.5]))
        self.weight_efficientnet = nn.Parameter(torch.FloatTensor([0.5]))
        
        # Feature alignment layers - unify to same size H×W×C
        self.feature_alignment = nn.ModuleDict({
            'inception': nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),  # Unify spatial size to 8x8
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            'efficientnet': nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),  # Unify spatial size to 8x8
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        })
        
        # Branch-level gating
        self.branch_gating = BranchGating(feature_dim=512)
        
        # Stage 1: Use residual feature fusion
        self.feature_fusion = ResidualFeatureFusion(feature_dim=512)
        # Original: self.feature_fusion = AdaptiveFeatureFusion(feature_dim=512, fusion_type='weighted')
        
        # Residual attention
        self.residual_attention = ResidualAttentionBlock(feature_dim=512)
        
        # Add projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        
        # Store class accuracies
        self.class_accuracies = None
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None, contrast_weight=0.1):
        # Update class accuracies
        if class_accuracies is not None:
            self.class_accuracies = class_accuracies
            
        if self.training and labels is not None:
            # Get Inception output
            inception_outputs = self.inception(x)
            if len(inception_outputs) == 3:
                inception_logits, inception_features, aux_logits = inception_outputs
            else:
                inception_logits, inception_features = inception_outputs
                aux_logits = None
            
            # Get EfficientNet output
            efficientnet_logits, efficientnet_features = self.efficientnet(x)
            
            # Align features to unified size H×W×C
            aligned_inception = self.feature_alignment['inception'](inception_features)
            aligned_efficientnet = self.feature_alignment['efficientnet'](efficientnet_features)
            
            # Branch-level gating
            gated_features = self.branch_gating(aligned_inception, aligned_efficientnet)
            
            # Adaptive feature fusion
            fused_features = self.feature_fusion(aligned_inception, aligned_efficientnet)
            
            # Residual attention
            attention_features = self.residual_attention(fused_features)
            
            # Weighted fusion
            weights = F.softmax(torch.stack([
                self.weight_inception,
                self.weight_efficientnet
            ]), dim=0)
            
            ensemble_logits = (
                weights[0] * inception_logits +
                weights[1] * efficientnet_logits
            )
            
            # Calculate losses for each model using aligned features
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
            
            # Use adaptive Focal Loss
            ce_loss = AdaptiveFocalLoss(num_classes=inception_logits.size(1), gamma=1.5, smoothing=0.2)
            if self.class_accuracies is not None:
                ce_loss.update_weights(self.class_accuracies)
            ensemble_loss = ce_loss(ensemble_logits, labels)
            
            # Calculate knowledge distillation loss
            kd_loss = self.kd_loss(efficientnet_logits, inception_logits, labels)
            
            # Generate contrastive learning features using attention-enhanced features
            inception_feat = F.adaptive_avg_pool2d(aligned_inception, (1, 1)).squeeze(-1).squeeze(-1)
            efficientnet_feat = F.adaptive_avg_pool2d(aligned_efficientnet, (1, 1)).squeeze(-1).squeeze(-1)
            attention_feat = F.adaptive_avg_pool2d(attention_features, (1, 1)).squeeze(-1).squeeze(-1)
            
            # Project features
            proj_inception = self.projection(inception_feat)
            proj_efficientnet = self.projection(efficientnet_feat)
            
            # Calculate contrastive learning and domain adaptation loss using attention-enhanced features
            contrast_domain_loss = self.adaptive_contrastive(
                features=self.projection(attention_feat),  # Use attention-enhanced features
                labels=labels,
                domain_features=torch.cat([inception_feat, efficientnet_feat, attention_feat], dim=0),
                alpha=alpha
            )
            
            # Total loss, use dynamic contrastive learning weight
            loss = (inception_loss + 
                   efficientnet_loss + 
                   ensemble_loss + 
                   0.5 * kd_loss + 
                   contrast_weight * 1.5 * contrast_domain_loss)  # Use dynamic contrastive learning weight
            
            # Add auxiliary loss
            if aux_logits is not None:
                aux_loss = ce_loss(aux_logits, labels)
                loss = loss + 0.4 * aux_loss
            
            return ensemble_logits, loss
        else:
            # Inference mode
            inception_logits, inception_features = self.inception(x)
            efficientnet_logits, efficientnet_features = self.efficientnet(x)
            
            # Align features to unified size H×W×C
            aligned_inception = self.feature_alignment['inception'](inception_features)
            aligned_efficientnet = self.feature_alignment['efficientnet'](efficientnet_features)
            
            # Branch-level gating
            gated_features = self.branch_gating(aligned_inception, aligned_efficientnet)
            
            # Adaptive feature fusion
            fused_features = self.feature_fusion(aligned_inception, aligned_efficientnet)
            
            # Residual attention
            attention_features = self.residual_attention(fused_features)
            
            # Weighted fusion of logits
            weights = F.softmax(torch.stack([
                self.weight_inception,
                self.weight_efficientnet
            ]), dim=0)
            
            ensemble_logits = (
                weights[0] * inception_logits +
                weights[1] * efficientnet_logits
            )
            
            # Create Grad-CAM instance, use last convolutional layer as target layer
            self.grad_cam = GradCAM(self, self.inception.inception.Mixed_7c) 
            
            return ensemble_logits, attention_features  # Return attention-enhanced features


class AdaptiveFeatureFusion(nn.Module):
    """Adaptive feature fusion module"""
    def __init__(self, feature_dim=512, fusion_type='weighted'):
        super(AdaptiveFeatureFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # Channel count after concatenation is 2x original
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'weighted':
            # Per-channel weighting
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
    """Residual attention module"""
    def __init__(self, feature_dim=512, reduction=16):
        super(ResidualAttentionBlock, self).__init__()
        
        # Channel attention: global average pooling → MLP → Sigmoid
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Lightweight spatial attention: depthwise 3x3 conv + Sigmoid
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Residual connection weights
        self.alpha = nn.Parameter(torch.tensor(0.3))  # Channel attention weight
        self.beta = nn.Parameter(torch.tensor(0.3))   # Spatial attention weight
    
    def forward(self, x):
        # Channel attention: y = x * s_c + x (residual)
        s_c = self.channel_attention(x)
        channel_refined = x * s_c + x  # Residual connection
        
        # Spatial attention: also residual
        s_s = self.spatial_attention(channel_refined)
        spatial_refined = channel_refined * s_s + channel_refined
        
        # Final residual fusion to prevent over-modulation
        return self.alpha * spatial_refined + (1 - self.alpha) * x


class BranchGating(nn.Module):
    """Branch-level gating mechanism"""
    def __init__(self, feature_dim=512):
        super(BranchGating, self).__init__()
        
        # Branch importance evaluation
        self.branch_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 2, 1),  # Output weights for two branches
            nn.Softmax(dim=1)
        )
        
        # Branch feature enhancement
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
        # Evaluate branch importance
        combined = torch.cat([f_inception, f_efficientnet], dim=1)
        branch_weights = self.branch_evaluator(combined)  # [B, 2, 1, 1]
        
        # Enhance each branch feature
        enhanced_inception = self.branch_enhancer['inception'](f_inception)
        enhanced_efficientnet = self.branch_enhancer['efficientnet'](f_efficientnet)
        
        # Gated fusion
        gated_inception = enhanced_inception * branch_weights[:, 0:1]
        gated_efficientnet = enhanced_efficientnet * branch_weights[:, 1:2]
        
        return gated_inception + gated_efficientnet 