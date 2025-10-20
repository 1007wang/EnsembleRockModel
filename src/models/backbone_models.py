import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .attention import EnhancedRockAttention, EfficientNetRockAttention, EfficientNetMinimalAttention, IdentityAttention
from .fpn import FPN, FeatureFusion
from .losses import EnhancedCombinedLoss
from .grad_cam import GradCAM

class ResNet50Model(nn.Module):
    def __init__(self, num_classes, use_attention=True):
        super(ResNet50Model, self).__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.channel_sizes = [256, 512, 1024, 2048]

        self.use_attention = use_attention
        if use_attention:
            self.enhanced_rock_attention = EnhancedRockAttention(in_channels=2048)

        self.fpn = FPN(
            in_channels_list=self.channel_sizes,
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

        for param in list(self.resnet.parameters())[:-100]:
            param.requires_grad = False

        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)

        self.grad_cam = GradCAM(self, self.resnet.layer4)

    def _extract_features(self, x):
        features = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)

        x = self.resnet.layer2(x)
        features.append(x)

        x = self.resnet.layer3(x)
        features.append(x)

        x = self.resnet.layer4(x)
        if self.use_attention:
            x = self.enhanced_rock_attention(x)
        features.append(x)

        return features

    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):

        features = self._extract_features(x)

        fpn_features = self.fpn(features)

        fused_features = self.fusion(fpn_features[0], fpn_features[-1])

        logits = self.fc(fused_features)

        if self.training and labels is not None:
            loss = self.criterion(
                fused_features,
                logits,
                labels,
                class_accuracies=class_accuracies,
            )
            return logits, loss

        return logits, fused_features

class EfficientNetB4Model(nn.Module):
    def __init__(self, num_classes, use_attention=True, attention_type='minimal'):
        super(EfficientNetB4Model, self).__init__()

        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        self.channel_sizes = [24, 56, 160, 1792]

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

                self.enhanced_rock_attention = EfficientNetMinimalAttention(in_channels=1792)

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

        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
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

        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)

        target_layer = self.efficientnet.features[-1]
        self.grad_cam = GradCAM(self, target_layer)

    def _extract_features(self, x):
        features = []
        current_feature = x

        layers = list(self.efficientnet.features)

        for layer in layers[:2]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[0](current_feature))

        for layer in layers[2:4]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[1](current_feature))

        for layer in layers[4:6]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[2](current_feature))

        for layer in layers[6:]:
            current_feature = layer(current_feature)

        if self.use_attention:
            current_feature = self.enhanced_rock_attention(current_feature)
        features.append(self.preprocess[3](current_feature))

        return features

    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):

        if x.size(2) != 380 or x.size(3) != 380:
            x = F.interpolate(x, size=(380, 380), mode='bilinear', align_corners=False)

        features = self._extract_features(x)

        fpn_features = self.fpn(features)

        fused_features = self.fusion(fpn_features[0], fpn_features[-1])

        logits = self.fc(fused_features)

        if self.training and labels is not None:
            loss = self.criterion(
                fused_features,
                logits,
                labels,
                class_accuracies=class_accuracies
            )
            return logits, loss

        return logits, fused_features

class InceptionV3Model(nn.Module):
    def __init__(self, num_classes, use_attention=True):
        super(InceptionV3Model, self).__init__()

        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )

        for param in list(self.inception.parameters())[:-150]:
            param.requires_grad = False

        self.use_attention = use_attention
        if use_attention:
            self.enhanced_rock_attention = EnhancedRockAttention(in_channels=2048)

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

        self.aux_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)

        self.grad_cam = GradCAM(self, self.inception.Mixed_7c)

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

        aux = x if self.training and self.inception.aux_logits else None

        x = self.inception.Mixed_7a(x)
        features.append(x)

        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)

        if self.use_attention:
            x = self.enhanced_rock_attention(x)
        features.append(x)

        return features, aux

    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):

        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        features, aux = self._extract_features(x)

        fpn_features = self.fpn(features)

        fused_features = self.fusion(fpn_features[0], fpn_features[-1])

        logits = self.fc(fused_features)

        if self.training and aux is not None:
            aux_logits = self.aux_fc(F.adaptive_avg_pool2d(aux, (1, 1)).view(aux.size(0), -1))
        else:
            aux_logits = None

        if self.training and labels is not None:

            main_loss = self.criterion(
                fused_features,
                logits,
                labels,
                class_accuracies=class_accuracies
            )

            if aux_logits is not None:
                aux_loss = 0.3 * F.cross_entropy(aux_logits, labels)
                loss = main_loss + aux_loss
            else:
                loss = main_loss

            return logits, loss

        return logits, fused_features

def create_model_without_attention(model_name, num_classes):
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

def create_model(model_name, num_classes, use_attention=True):
    if model_name == 'resnet50':
        return ResNet50Model(num_classes, use_attention)
    elif model_name == 'resnet50_optimized':
        return OptimizedResNet50Model(num_classes, use_attention)
    elif model_name == 'efficientnet_b4':
        return EfficientNetB4Model(num_classes, use_attention)
    elif model_name == 'inceptionv3':
        return InceptionV3Model(num_classes, use_attention)
    elif model_name == 'ensemble':

        from .ensemble_model import EnsembleModel
        return EnsembleModel(num_classes)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

class OptimizedResNet50Model(nn.Module):
    def __init__(self, num_classes, use_attention=True):
        super(OptimizedResNet50Model, self).__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.channel_sizes = [256, 512, 1024, 2048]

        self.use_attention = use_attention
        if use_attention:
            from .attention import ResNet50EnhancedAttention
            self.enhanced_attention = ResNet50EnhancedAttention(in_channels=2048)

            self.mid_attention_1024 = ResNet50EnhancedAttention(in_channels=1024)
            self.mid_attention_512 = ResNet50EnhancedAttention(in_channels=512)

        self.fpn = FPN(
            in_channels_list=self.channel_sizes,
            out_channels=512
        )

        self.multi_level_fusion = nn.Sequential(
            nn.Conv2d(512 * 4, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(256, num_classes)
        )

        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self._freeze_layers()

        from .losses import ResNet50OptimizedLoss
        self.criterion = ResNet50OptimizedLoss(num_classes=num_classes, feat_dim=512)

        self.grad_cam = GradCAM(self, self.resnet.layer4)

    def _freeze_layers(self):

        layers_to_freeze = [
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.layer1,

            *list(self.resnet.layer2.children())[:2],
        ]

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def _extract_features(self, x):
        features = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)

        x = self.resnet.layer2(x)
        if self.use_attention:
            x = self.mid_attention_512(x)
        features.append(x)

        x = self.resnet.layer3(x)
        if self.use_attention:
            x = self.mid_attention_1024(x)
        features.append(x)

        x = self.resnet.layer4(x)
        if self.use_attention:
            x = self.enhanced_attention(x)
        features.append(x)

        return features

    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        batch_size = x.size(0)

        features = self._extract_features(x)

        fpn_features = self.fpn(features)

        target_size = fpn_features[0].shape[2:]
        aligned_features = []

        for feat in fpn_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        fused_features = torch.cat(aligned_features, dim=1)
        fused_features = self.multi_level_fusion(fused_features)

        logits = self.classifier(fused_features)

        aux_logits = None
        if self.training:
            aux_logits = self.aux_classifier(features[2])

        if self.training and labels is not None:

            main_loss = self.criterion(fused_features, logits, labels, class_accuracies)

            aux_loss = 0
            if aux_logits is not None:
                aux_loss = F.cross_entropy(aux_logits, labels) * 0.3

            total_loss = main_loss + aux_loss
            return logits, total_loss

        return logits, fused_features 
