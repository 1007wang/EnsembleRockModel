import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCombinedLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=512):
        super(EnhancedCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.focal_loss = AdaptiveFocalLoss(num_classes, gamma=1.5, smoothing=0.2)

        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.metric_fc = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.xavier_normal_(self.metric_fc.weight, gain=0.02)

        self.temperature = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, features, logits, labels, class_accuracies=None):

        features = torch.clamp(features, -10, 10)
        logits = torch.clamp(logits, -10, 10)

        processed_features = self.feature_processor(features)

        metric_logits = self.metric_fc(processed_features)

        if class_accuracies is not None:
            self.focal_loss.update_weights(class_accuracies)

        metric_loss = self.focal_loss(metric_logits, labels)

        cls_loss = self.focal_loss(logits, labels)

        contrast_loss = self.contrastive_loss(processed_features, labels)

        total_loss = cls_loss + 0.3 * metric_loss + 0.1 * contrast_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return cls_loss

        return total_loss

    def contrastive_loss(self, features, labels):

        features = F.normalize(features, dim=1, eps=1e-8)
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = torch.clamp(similarity_matrix, -1, 1)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        similarity_matrix = similarity_matrix / self.temperature
        exp_sim = F.log_softmax(similarity_matrix, dim=1)

        mask_self = torch.eye(mask.size(0), device=mask.device)
        mask = mask * (1 - mask_self)

        pos_sim = similarity_matrix[mask.bool()]
        neg_sim = similarity_matrix[(1 - mask).bool()]

        if len(pos_sim) == 0:
            return torch.tensor(0.0, device=features.device)

        loss = -torch.mean(pos_sim) + torch.logsumexp(neg_sim, dim=0)
        return torch.clamp(loss, min=0.0, max=10.0)

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        hard_loss = self.ce(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=1.5, smoothing=0.2):
        super(AdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer('class_weights', None)

    def update_weights(self, class_accuracies):

        weights = torch.tensor([1.0 / (acc ** 2 + 0.05) for acc in class_accuracies.values()])

        weights = weights / weights.sum()

        weights = torch.clamp(weights, min=0.1, max=10.0)

        weights = weights / weights.sum()

        self.class_weights = weights

    def forward(self, pred, target):
        eps = 1e-7
        pred = F.softmax(pred, dim=1)

        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes

        pt = (smooth_one_hot * pred).sum(1) + eps
        focal_weight = (1 - pt) ** self.gamma

        if self.class_weights is not None:

            class_weight = self.class_weights.to(target.device)[target]
            focal_weight = focal_weight * class_weight

        loss = -torch.log(pt) * focal_weight

        loss = torch.clamp(loss, max=100.0)

        return loss.mean()

class HardSampleMiningLoss(nn.Module):
    def __init__(self, num_classes, mining_ratio=0.3, margin=0.5):
        super(HardSampleMiningLoss, self).__init__()
        self.num_classes = num_classes
        self.mining_ratio = mining_ratio
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):

        losses = self.ce_loss(logits, labels)

        probs = F.softmax(logits, dim=1)

        confidence = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()

        batch_size = logits.size(0)
        num_hard = int(batch_size * self.mining_ratio)

        hard_indices = torch.argsort(confidence)[:num_hard]

        weighted_losses = losses.clone()
        weighted_losses[hard_indices] *= 2.0

        return weighted_losses.mean()

class TriangularLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TriangularLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):

        features = F.normalize(features, p=2, dim=1)

        intra_class_dist = self._compute_intra_class_distance(features, labels)

        inter_class_dist = self._compute_inter_class_distance(features, labels)

        loss = F.relu(intra_class_dist - inter_class_dist + self.margin)

        return loss.mean()

    def _compute_intra_class_distance(self, features, labels):
        unique_labels = torch.unique(labels)
        intra_distances = []

        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 1:
                class_features = features[mask]

                pairwise_dist = torch.cdist(class_features, class_features, p=2)

                mask_diag = ~torch.eye(pairwise_dist.size(0), dtype=bool, device=pairwise_dist.device)
                intra_distances.append(pairwise_dist[mask_diag].mean())

        return torch.stack(intra_distances).mean() if intra_distances else torch.tensor(0.0, device=features.device)

    def _compute_inter_class_distance(self, features, labels):
        unique_labels = torch.unique(labels)
        inter_distances = []

        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                mask1 = labels == label1
                mask2 = labels == label2

                features1 = features[mask1]
                features2 = features[mask2]

                pairwise_dist = torch.cdist(features1, features2, p=2)
                inter_distances.append(pairwise_dist.mean())

        return torch.stack(inter_distances).mean() if inter_distances else torch.tensor(1.0, device=features.device)

class ResNet50OptimizedLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=512):
        super(ResNet50OptimizedLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.focal_loss = AdaptiveFocalLoss(num_classes, gamma=1.2, smoothing=0.15)
        self.hard_mining_loss = HardSampleMiningLoss(num_classes, mining_ratio=0.25)
        self.triangular_loss = TriangularLoss(margin=0.8)

        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15)
        )

        self.aux_classifier = nn.Linear(feat_dim, num_classes)

        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3, 0.2]))

    def forward(self, features, logits, labels, class_accuracies=None):

        processed_features = self.feature_processor(features)

        aux_logits = self.aux_classifier(processed_features)

        if class_accuracies is not None:
            self.focal_loss.update_weights(class_accuracies)

        focal_loss = self.focal_loss(logits, labels)
        aux_focal_loss = self.focal_loss(aux_logits, labels)
        hard_mining_loss = self.hard_mining_loss(logits, labels)
        triangular_loss = self.triangular_loss(processed_features, labels)

        weights = F.softmax(self.loss_weights, dim=0)

        total_loss = (weights[0] * focal_loss + 
                     weights[1] * aux_focal_loss + 
                     weights[2] * hard_mining_loss + 
                     weights[3] * triangular_loss)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return focal_loss

        return total_loss 
