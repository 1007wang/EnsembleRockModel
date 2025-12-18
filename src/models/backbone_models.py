import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .attention import EnhancedRockAttention, EfficientNetRockAttention, EfficientNetMinimalAttention, IdentityAttention
from .fpn import FPN, FeatureFusion
from .losses import EnhancedCombinedLoss
from .grad_cam import GradCAM



class ResNet50Model(nn.Module):
    """Rock classification model using ResNet50 as backbone network"""
    def __init__(self, num_classes, use_attention=True):
        super(ResNet50Model, self).__init__()
        # Use pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Feature layer channel sizes
        self.channel_sizes = [256, 512, 1024, 2048]
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.enhanced_rock_attention = EnhancedRockAttention(in_channels=2048)
        
        # Feature Pyramid Network
        self.fpn = FPN(
            in_channels_list=self.channel_sizes,
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
        for param in list(self.resnet.parameters())[:-100]:
            param.requires_grad = False
            
        # Loss function
        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)
        
        # GradCAM visualization - use last convolutional layer as target layer
        self.grad_cam = GradCAM(self, self.resnet.layer4)
    
    def _extract_features(self, x):
        """Extract features from each stage of ResNet50"""
        features = []
        
        # Stage 1: Initial convolution and pooling
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # Stages 2-5
        x = self.resnet.layer1(x)  # 256 channels
        features.append(x)
        
        x = self.resnet.layer2(x)  # 512 channels
        features.append(x)
        
        x = self.resnet.layer3(x)  # 1024 channels
        features.append(x)
        
        x = self.resnet.layer4(x)  # 2048 channels
        if self.use_attention:
            x = self.enhanced_rock_attention(x)
        features.append(x)
        
        return features
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        # Extract features
        features = self._extract_features(x)
        
        # Feature pyramid enhancement
        fpn_features = self.fpn(features)
        
        # Feature fusion
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # Classification
        logits = self.fc(fused_features)
        
        # If in training mode and labels provided, calculate loss
        if self.training and labels is not None:
            loss = self.criterion(
                fused_features,  # Features
                logits,          # Classification output
                labels,          # Labels
                class_accuracies=class_accuracies,
            )
            return logits, loss
        
        return logits, fused_features

class EfficientNetB4Model(nn.Module):
    """Rock classification model using EfficientNetB4 as backbone network"""
    def __init__(self, num_classes, use_attention=True, attention_type='minimal'):
        super(EfficientNetB4Model, self).__init__()
        # Use pre-trained EfficientNetB4
        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # EfficientNetB4 feature layer channel sizes
        self.channel_sizes = [24, 56, 160, 1792]
        
        # Attention mechanism - support multiple attention types
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
                # Default to most conservative version
                self.enhanced_rock_attention = EfficientNetMinimalAttention(in_channels=1792)
        
        # Feature preprocessing layers (unify spatial dimensions and channel counts)
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
        
        # Feature Pyramid Network
        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
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
            
        # Loss function
        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)
        
        # GradCAM visualization - use last feature layer as target layer
        # EfficientNet's feature extractor is usually a Sequential, get the last layer
        target_layer = self.efficientnet.features[-1]
        self.grad_cam = GradCAM(self, target_layer)
    
    def _extract_features(self, x):
        """Extract intermediate features from EfficientNetB4"""
        features = []
        current_feature = x
        
        # Get all layers from feature extractor
        layers = list(self.efficientnet.features)
        
        # Stage 1: 24 channels
        for layer in layers[:2]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[0](current_feature))
        
        # Stage 2: 56 channels
        for layer in layers[2:4]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[1](current_feature))
        
        # Stage 3: 160 channels
        for layer in layers[4:6]:
            current_feature = layer(current_feature)
        features.append(self.preprocess[2](current_feature))
        
        # Stage 4: 1792 channels
        for layer in layers[6:]:
            current_feature = layer(current_feature)
            
        if self.use_attention:
            current_feature = self.enhanced_rock_attention(current_feature)
        features.append(self.preprocess[3](current_feature))
        
        return features
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        # Ensure input size is correct (EfficientNet-B4 optimal input size is 380x380)
        if x.size(2) != 380 or x.size(3) != 380:
            x = F.interpolate(x, size=(380, 380), mode='bilinear', align_corners=False)
            
        # Extract features
        features = self._extract_features(x)
        
        # Feature pyramid enhancement
        fpn_features = self.fpn(features)
        
        # Feature fusion
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # Classification
        logits = self.fc(fused_features)
        
        # If in training mode and labels provided, calculate loss
        if self.training and labels is not None:
            loss = self.criterion(
                fused_features,  # Features
                logits,          # Classification output
                labels,          # Labels
                class_accuracies=class_accuracies
            )
            return logits, loss
        
        return logits, fused_features

class InceptionV3Model(nn.Module):
    """Rock classification model using InceptionV3 as backbone network"""
    def __init__(self, num_classes, use_attention=True):
        super(InceptionV3Model, self).__init__()
        # Use pre-trained Inception V3
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
        
        # Freeze some layers
        for param in list(self.inception.parameters())[:-150]:
            param.requires_grad = False
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.enhanced_rock_attention = EnhancedRockAttention(in_channels=2048)
        
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
        
        # Auxiliary classifier
        self.aux_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Loss function
        self.criterion = EnhancedCombinedLoss(num_classes=num_classes)
        
        # GradCAM visualization - use last mixed layer as target layer
        self.grad_cam = GradCAM(self, self.inception.Mixed_7c)
    
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
        aux = x if self.training and self.inception.aux_logits else None
        
        # Stage 4: Mixed_7a (1280 channels)
        x = self.inception.Mixed_7a(x)  # 1280
        features.append(x)  # 1280 channels
        
        # Stage 5: Mixed_7b/c (2048 channels)
        x = self.inception.Mixed_7b(x)  # 2048
        x = self.inception.Mixed_7c(x)  # 2048
        
        # Apply attention mechanism
        if self.use_attention:
            x = self.enhanced_rock_attention(x)
        features.append(x)  # 2048 channels
        
        return features, aux
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        # Ensure input size is correct (InceptionV3 requires 299x299)
        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
        # Extract features
        features, aux = self._extract_features(x)
        
        # Feature pyramid enhancement
        fpn_features = self.fpn(features)
        
        # Feature fusion
        fused_features = self.fusion(fpn_features[0], fpn_features[-1])
        
        # Classification
        logits = self.fc(fused_features)
        
        # Auxiliary classifier (if in training phase)
        if self.training and aux is not None:
            aux_logits = self.aux_fc(F.adaptive_avg_pool2d(aux, (1, 1)).view(aux.size(0), -1))
        else:
            aux_logits = None
        
        # If in training mode and labels provided, calculate loss
        if self.training and labels is not None:
            # Main classifier loss
            main_loss = self.criterion(
                fused_features,  # Features
                logits,          # Classification output
                labels,          # Labels
                class_accuracies=class_accuracies
            )
            
            # Auxiliary classifier loss (if auxiliary classifier exists)
            if aux_logits is not None:
                aux_loss = 0.3 * F.cross_entropy(aux_logits, labels)
                loss = main_loss + aux_loss
            else:
                loss = main_loss
                
            return logits, loss
        
        return logits, fused_features

# Model versions without attention mechanism
def create_model_without_attention(model_name, num_classes):
    """Create model without attention mechanism"""
    if model_name == 'resnet50':
        return ResNet50Model(num_classes, use_attention=False)
    elif model_name == 'resnet50_optimized':
        return OptimizedResNet50Model(num_classes, use_attention=False)
    elif model_name == 'efficientnet_b4':
        return EfficientNetB4Model(num_classes, use_attention=False)
    elif model_name == 'inceptionv3':
        return InceptionV3Model(num_classes, use_attention=False)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

# Model creation function
def create_model(model_name, num_classes, use_attention=True):
    """Create model of specified type"""
    if model_name == 'resnet50':
        return ResNet50Model(num_classes, use_attention)
    elif model_name == 'resnet50_optimized':
        return OptimizedResNet50Model(num_classes, use_attention)
    elif model_name == 'efficientnet_b4':
        return EfficientNetB4Model(num_classes, use_attention)
    elif model_name == 'inceptionv3':
        return InceptionV3Model(num_classes, use_attention)
    elif model_name == 'ensemble':
        # Import EnsembleModel and return
        from .ensemble_model import EnsembleModel
        return EnsembleModel(num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

class OptimizedResNet50Model(nn.Module):
    """Optimized ResNet50 model - specifically optimized for rock classification task"""
    def __init__(self, num_classes, use_attention=True):
        super(OptimizedResNet50Model, self).__init__()
        
        # Use pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Feature layer channel sizes
        self.channel_sizes = [256, 512, 1024, 2048]
        
        # Attention mechanism - use new optimized attention
        self.use_attention = use_attention
        if use_attention:
            from .attention import ResNet50EnhancedAttention
            self.enhanced_attention = ResNet50EnhancedAttention(in_channels=2048)
            
            # Also add attention to intermediate layers (optional)
            self.mid_attention_1024 = ResNet50EnhancedAttention(in_channels=1024)
            self.mid_attention_512 = ResNet50EnhancedAttention(in_channels=512)
        
        # Improved Feature Pyramid Network
        self.fpn = FPN(
            in_channels_list=self.channel_sizes,
            out_channels=512
        )
        
        # Multi-level feature fusion
        self.multi_level_fusion = nn.Sequential(
            nn.Conv2d(512 * 4, 1024, 1),  # Fuse features from 4 levels
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            # Global average pooling + global max pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # First layer
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            
            # Second layer
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            
            # Third layer (auxiliary feature extraction)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            
            # Output layer
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary classification head (for deep supervision)
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),  # 1024 channels from layer3
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freezing strategy - more fine-grained freezing
        self._freeze_layers()
        
        # Use optimized loss function
        from .losses import ResNet50OptimizedLoss
        self.criterion = ResNet50OptimizedLoss(num_classes=num_classes, feat_dim=512)
        
        # GradCAM visualization
        self.grad_cam = GradCAM(self, self.resnet.layer4)
    
    def _freeze_layers(self):
        """Fine-grained layer freezing strategy"""
        # Freeze early layers, preserve trainability of later layers
        layers_to_freeze = [
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.layer1,
            # Freeze first half of layer2
            *list(self.resnet.layer2.children())[:2],
        ]
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def _extract_features(self, x):
        """Extract features from each stage of ResNet50"""
        features = []
        
        # Stage 1: Initial convolution and pooling
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # Stages 2-5, apply attention at key layers
        x = self.resnet.layer1(x)  # 256 channels
        features.append(x)
        
        x = self.resnet.layer2(x)  # 512 channels
        if self.use_attention:
            x = self.mid_attention_512(x)
        features.append(x)
        
        x = self.resnet.layer3(x)  # 1024 channels
        if self.use_attention:
            x = self.mid_attention_1024(x)
        features.append(x)
        
        x = self.resnet.layer4(x)  # 2048 channels
        if self.use_attention:
            x = self.enhanced_attention(x)
        features.append(x)
        
        return features
    
    def forward(self, x, labels=None, alpha=1.0, class_accuracies=None):
        batch_size = x.size(0)
        
        # Extract multi-level features
        features = self._extract_features(x)
        
        # Feature pyramid enhancement
        fpn_features = self.fpn(features)
        
        # Multi-level feature fusion
        # Adjust all FPN features to same size and concatenate
        target_size = fpn_features[0].shape[2:]
        aligned_features = []
        
        for feat in fpn_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # Concatenate all features
        fused_features = torch.cat(aligned_features, dim=1)
        fused_features = self.multi_level_fusion(fused_features)
        
        # Main classification
        logits = self.classifier(fused_features)
        
        # Auxiliary classification (using layer3 features)
        aux_logits = None
        if self.training:
            aux_logits = self.aux_classifier(features[2])  # layer3 features
        
        # If in training mode and labels provided, calculate loss
        if self.training and labels is not None:
            # Main loss
            main_loss = self.criterion(fused_features, logits, labels, class_accuracies)
            
            # Auxiliary loss
            aux_loss = 0
            if aux_logits is not None:
                aux_loss = F.cross_entropy(aux_logits, labels) * 0.3
            
            total_loss = main_loss + aux_loss
            return logits, total_loss
        
        return logits, fused_features 