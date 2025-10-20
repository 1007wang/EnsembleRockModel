import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .fpn import FPN, FeatureFusion
from .losses import KnowledgeDistillationLoss, EnhancedCombinedLoss, AdaptiveFocalLoss
from .contrastive_losses import CombinedAdaptiveContrastiveLoss
from .grad_cam import GradCAM

class EnhancedInceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedInceptionV3, self).__init__()
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )

        for param in list(self.inception.parameters())[:-150]:
            param.requires_grad = False

        self.fpn = FPN(
            in_channels_list=[288, 768, 1280, 2048],
            out_channels=512
        )

        self.fusion = FeatureFusion(channels=512)

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
        features = []

        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        features.append(x)

        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        features.append(x)

        aux = None
        if self.training and self.inception.aux_logits:
            aux = self.inception.AuxLogits(x)

        x = self.inception.Mixed_7a(x)
        features.append(x)

        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        features.append(x)

        return features, aux

    def forward(self, x):

        features, aux = self._extract_features(x)

        fpn_features = self.fpn(features)

        fused_features = self.fusion(fpn_features[0], fpn_features[-1])

        x = self.fc(fused_features)

        if self.training and aux is not None:
            return x, fused_features, aux
        return x, fused_features

class EfficientNetB4Enhanced(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4Enhanced, self).__init__()
        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        self.channel_sizes = [24, 56, 160, 1792]

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

        self.fpn = FPN(
            in_channels_list=[288, 768, 1280, 2048],
            out_channels=512
        )

        self.fusion = FeatureFusion(channels=512)

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

        for param in list(self.efficientnet.parameters())[:-100]:
            param.requires_grad = False

    def _extract_features(self, x):
        features = []
        current_feature = x

        layers = list(self.efficientnet.features)

        for layer in layers[:2]:
            current_feature = layer(current_feature)
        features.append(current_feature)

        for layer in layers[2:4]:
            current_feature = layer(current_feature)
        features.append(current_feature)

        for layer in layers[4:6]:
            current_feature = layer(current_feature)
        features.append(current_feature)

        for layer in layers[6:]:
            current_feature = layer(current_feature)
        features.append(current_feature)

        processed_features = []
        for feat, preprocess in zip(features, self.preprocess):
            processed_features.append(preprocess(feat))

        return processed_features

    def forward(self, x):

        if x.shape[-1] != 380:
            x = F.interpolate(x, size=(380, 380), mode='bilinear', align_corners=True)

        features = self._extract_features(x)

        fpn_features = self.fpn(features)

        fused_features = self.fusion(fpn_features[0], fpn_features[-1])

        x = self.fc(fused_features)

        return x, fused_features

class EnsembleModel(nn.Module):
    def __init__(self, num_classes, temperature=4.0):
        super(EnsembleModel, self).__init__()
        self.inception = EnhancedInceptionV3(num_classes)
        self.efficientnet = EfficientNetB4Enhanced(num_classes)

        self.combined_loss = EnhancedCombinedLoss(num_classes, feat_dim=512)
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature)
        self.ce = nn.CrossEntropyLoss()

        self.adaptive_contrastive = CombinedAdaptiveContrastiveLoss(
            feature_dim=512,
            num_classes=num_classes,
            temperature=0.07,
            memory_bank_size=4096
        )

        self.grad_cam = None

        self.weight_inception = nn.Parameter(torch.FloatTensor([0.5]))
        self.weight_efficientnet = nn.Parameter(torch.FloatTensor([0.5]))

        self.feature_alignment = nn.ModuleDict({
            'inception': nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            'efficientnet': nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        })

        self.branch_gating = BranchGating(feature_dim=512)

        self.feature_fusion = AdaptiveFeatureFusion(feature_dim=512, fusion_type='weighted')

        self.residual_attention = ResidualAttentionBlock(feature_dim=512)

        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

        self.class_accuracies = None

    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None, contrast_weight=0.1):

        if class_accuracies is not None:
            self.class_accuracies = class_accuracies

        if self.training and labels is not None:

            inception_outputs = self.inception(x)
            if len(inception_outputs) == 3:
                inception_logits, inception_features, aux_logits = inception_outputs
            else:
                inception_logits, inception_features = inception_outputs
                aux_logits = None

            efficientnet_logits, efficientnet_features = self.efficientnet(x)

            aligned_inception = self.feature_alignment['inception'](inception_features)
            aligned_efficientnet = self.feature_alignment['efficientnet'](efficientnet_features)

            gated_features = self.branch_gating(aligned_inception, aligned_efficientnet)

            fused_features = self.feature_fusion(aligned_inception, aligned_efficientnet)

            attention_features = self.residual_attention(fused_features)

            weights = F.softmax(torch.stack([
                self.weight_inception,
                self.weight_efficientnet
            ]), dim=0)

            ensemble_logits = (
                weights[0] * inception_logits +
                weights[1] * efficientnet_logits
            )

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

            ce_loss = AdaptiveFocalLoss(num_classes=inception_logits.size(1), gamma=1.5, smoothing=0.2)
            if self.class_accuracies is not None:
                ce_loss.update_weights(self.class_accuracies)
            ensemble_loss = ce_loss(ensemble_logits, labels)

            kd_loss = self.kd_loss(efficientnet_logits, inception_logits, labels)

            inception_feat = F.adaptive_avg_pool2d(aligned_inception, (1, 1)).squeeze(-1).squeeze(-1)
            efficientnet_feat = F.adaptive_avg_pool2d(aligned_efficientnet, (1, 1)).squeeze(-1).squeeze(-1)
            attention_feat = F.adaptive_avg_pool2d(attention_features, (1, 1)).squeeze(-1).squeeze(-1)

            proj_inception = self.projection(inception_feat)
            proj_efficientnet = self.projection(efficientnet_feat)

            contrast_domain_loss = self.adaptive_contrastive(
                features=self.projection(attention_feat),
                labels=labels,
                domain_features=torch.cat([inception_feat, efficientnet_feat, attention_feat], dim=0),
                alpha=alpha
            )

            loss = (inception_loss + 
                   efficientnet_loss + 
                   ensemble_loss + 
                   0.5 * kd_loss + 
                   contrast_weight * 1.5 * contrast_domain_loss)

            if aux_logits is not None:
                aux_loss = ce_loss(aux_logits, labels)
                loss = loss + 0.4 * aux_loss

            return ensemble_logits, loss
        else:

            inception_logits, inception_features = self.inception(x)
            efficientnet_logits, efficientnet_features = self.efficientnet(x)

            aligned_inception = self.feature_alignment['inception'](inception_features)
            aligned_efficientnet = self.feature_alignment['efficientnet'](efficientnet_features)

            gated_features = self.branch_gating(aligned_inception, aligned_efficientnet)

            fused_features = self.feature_fusion(aligned_inception, aligned_efficientnet)

            attention_features = self.residual_attention(fused_features)

            weights = F.softmax(torch.stack([
                self.weight_inception,
                self.weight_efficientnet
            ]), dim=0)

            ensemble_logits = (
                weights[0] * inception_logits +
                weights[1] * efficientnet_logits
            )

            self.grad_cam = GradCAM(self, self.inception.inception.Mixed_7c) 

            return ensemble_logits, attention_features

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, feature_dim=512, fusion_type='weighted'):
        super(AdaptiveFeatureFusion, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':

            self.fusion_conv = nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'weighted':

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
    def __init__(self, feature_dim=512, reduction=16):
        super(ResidualAttentionBlock, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, feature_dim, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):

        s_c = self.channel_attention(x)
        channel_refined = x * s_c + x

        s_s = self.spatial_attention(channel_refined)
        spatial_refined = channel_refined * s_s + channel_refined

        return self.alpha * spatial_refined + (1 - self.alpha) * x

class BranchGating(nn.Module):
    def __init__(self, feature_dim=512):
        super(BranchGating, self).__init__()

        self.branch_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 2, 1),
            nn.Softmax(dim=1)
        )

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

        combined = torch.cat([f_inception, f_efficientnet], dim=1)
        branch_weights = self.branch_evaluator(combined)

        enhanced_inception = self.branch_enhancer['inception'](f_inception)
        enhanced_efficientnet = self.branch_enhancer['efficientnet'](f_efficientnet)

        gated_inception = enhanced_inception * branch_weights[:, 0:1]
        gated_efficientnet = enhanced_efficientnet * branch_weights[:, 1:2]

        return gated_inception + gated_efficientnet 
