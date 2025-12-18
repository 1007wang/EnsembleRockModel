import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCombinedLoss(nn.Module):
    """增强版组合损失"""
    def __init__(self, num_classes, feat_dim=512):
        super(EnhancedCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # 使用更稳定的Focal Loss，使用调整后的参数
        self.focal_loss = AdaptiveFocalLoss(num_classes, gamma=1.5, smoothing=0.2)
        
        # 改进特征处理
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)  # 添加dropout
        )
        
        # 改进度量学习头
        self.metric_fc = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.xavier_normal_(self.metric_fc.weight, gain=0.02)  # 降低初始化权重
        
        # 对比学习温度参数
        self.temperature = nn.Parameter(torch.FloatTensor([0.1]))  # 提高初始温度
        
    def forward(self, features, logits, labels, class_accuracies=None):
        # 梯度裁剪
        features = torch.clamp(features, -10, 10)
        logits = torch.clamp(logits, -10, 10)
        
        # 处理特征
        processed_features = self.feature_processor(features)
        
        # 计算度量损失
        metric_logits = self.metric_fc(processed_features)
        
        # 更新类别权重
        if class_accuracies is not None:
            self.focal_loss.update_weights(class_accuracies)
            
        metric_loss = self.focal_loss(metric_logits, labels)
        
        # 计算分类损失
        cls_loss = self.focal_loss(logits, labels)
        
        # 计算对比损失（添加数值稳定性）
        contrast_loss = self.contrastive_loss(processed_features, labels)
        
        # 使用动态权重，增加对比损失的权重
        total_loss = cls_loss + 0.3 * metric_loss + 0.1 * contrast_loss
        
        # 检查并处理无效值
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return cls_loss  # 如果组合损失无效，只返回分类损失
            
        return total_loss
    
    def contrastive_loss(self, features, labels):
        # 添加数值稳定性的对比损失计算
        features = F.normalize(features, dim=1, eps=1e-8)
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = torch.clamp(similarity_matrix, -1, 1)  # 限制相似度范围
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 使用log_softmax进行数值稳定的计算
        similarity_matrix = similarity_matrix / self.temperature
        exp_sim = F.log_softmax(similarity_matrix, dim=1)
        
        # 去除自身相似度
        mask_self = torch.eye(mask.size(0), device=mask.device)
        mask = mask * (1 - mask_self)
        
        # 计算正样本对的损失
        pos_sim = similarity_matrix[mask.bool()]
        neg_sim = similarity_matrix[(1 - mask).bool()]
        
        if len(pos_sim) == 0:  # 处理边界情况
            return torch.tensor(0.0, device=features.device)
            
        loss = -torch.mean(pos_sim) + torch.logsumexp(neg_sim, dim=0)
        return torch.clamp(loss, min=0.0, max=10.0)  # 限制损失范围

class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失"""
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
    """自适应Focal Loss"""
    def __init__(self, num_classes, gamma=1.5, smoothing=0.2):
        super(AdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma  # 降低gamma值，减少对易分类样本的惩罚
        self.smoothing = smoothing  # 增加标签平滑，提高泛化能力
        self.register_buffer('class_weights', None)
        
    def update_weights(self, class_accuracies):
        """根据类别准确率动态更新权重"""
        # 使用更激进的权重调整策略
        weights = torch.tensor([1.0 / (acc ** 2 + 0.05) for acc in class_accuracies.values()])
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 限制权重范围，避免某些类别权重过大
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        # 再次归一化
        weights = weights / weights.sum()
        
        self.class_weights = weights  # 不指定设备，让register_buffer自动处理
    
    def forward(self, pred, target):
        eps = 1e-7
        pred = F.softmax(pred, dim=1)
        
        # 创建平滑标签
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        # 计算focal loss
        pt = (smooth_one_hot * pred).sum(1) + eps
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用类别权重
        if self.class_weights is not None:
            # 确保class_weights与target在同一设备上
            class_weight = self.class_weights.to(target.device)[target]
            focal_weight = focal_weight * class_weight
        
        loss = -torch.log(pt) * focal_weight
        
        # 添加数值稳定性检查
        loss = torch.clamp(loss, max=100.0)  # 限制损失值上限
        
        return loss.mean()

class HardSampleMiningLoss(nn.Module):
    """困难样本挖掘损失 - 专门针对岩石分类中的困难样本"""
    def __init__(self, num_classes, mining_ratio=0.3, margin=0.5):
        super(HardSampleMiningLoss, self).__init__()
        self.num_classes = num_classes
        self.mining_ratio = mining_ratio  # 困难样本占比
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, labels):
        # 计算每个样本的损失
        losses = self.ce_loss(logits, labels)
        
        # 获取预测概率
        probs = F.softmax(logits, dim=1)
        
        # 计算预测置信度
        confidence = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()
        
        # 选择困难样本（低置信度样本）
        batch_size = logits.size(0)
        num_hard = int(batch_size * self.mining_ratio)
        
        # 基于置信度选择困难样本
        hard_indices = torch.argsort(confidence)[:num_hard]
        
        # 对困难样本给予更高权重
        weighted_losses = losses.clone()
        weighted_losses[hard_indices] *= 2.0
        
        return weighted_losses.mean()

class TriangularLoss(nn.Module):
    """三角损失 - 增强类间距离"""
    def __init__(self, margin=1.0):
        super(TriangularLoss, self).__init__()
        self.margin = margin
        
    def forward(self, features, labels):
        # 特征归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 计算类内距离（应该小）
        intra_class_dist = self._compute_intra_class_distance(features, labels)
        
        # 计算类间距离（应该大）
        inter_class_dist = self._compute_inter_class_distance(features, labels)
        
        # 三角不等式损失
        loss = F.relu(intra_class_dist - inter_class_dist + self.margin)
        
        return loss.mean()
    
    def _compute_intra_class_distance(self, features, labels):
        """计算类内平均距离"""
        unique_labels = torch.unique(labels)
        intra_distances = []
        
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 1:
                class_features = features[mask]
                # 计算该类内所有样本对的距离
                pairwise_dist = torch.cdist(class_features, class_features, p=2)
                # 去除对角线（自身距离为0）
                mask_diag = ~torch.eye(pairwise_dist.size(0), dtype=bool, device=pairwise_dist.device)
                intra_distances.append(pairwise_dist[mask_diag].mean())
        
        return torch.stack(intra_distances).mean() if intra_distances else torch.tensor(0.0, device=features.device)
    
    def _compute_inter_class_distance(self, features, labels):
        """计算类间平均距离"""
        unique_labels = torch.unique(labels)
        inter_distances = []
        
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                mask1 = labels == label1
                mask2 = labels == label2
                
                features1 = features[mask1]
                features2 = features[mask2]
                
                # 计算两类间的距离
                pairwise_dist = torch.cdist(features1, features2, p=2)
                inter_distances.append(pairwise_dist.mean())
        
        return torch.stack(inter_distances).mean() if inter_distances else torch.tensor(1.0, device=features.device)

class ResNet50OptimizedLoss(nn.Module):
    """专门为ResNet50优化的综合损失函数"""
    def __init__(self, num_classes, feat_dim=512):
        super(ResNet50OptimizedLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # 主要损失组件
        self.focal_loss = AdaptiveFocalLoss(num_classes, gamma=1.2, smoothing=0.15)
        self.hard_mining_loss = HardSampleMiningLoss(num_classes, mining_ratio=0.25)
        self.triangular_loss = TriangularLoss(margin=0.8)
        
        # 特征处理
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15)
        )
        
        # 辅助分类头
        self.aux_classifier = nn.Linear(feat_dim, num_classes)
        
        # 损失权重（可学习）
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3, 0.2]))
        
    def forward(self, features, logits, labels, class_accuracies=None):
        # 处理特征
        processed_features = self.feature_processor(features)
        
        # 辅助分类
        aux_logits = self.aux_classifier(processed_features)
        
        # 更新自适应权重
        if class_accuracies is not None:
            self.focal_loss.update_weights(class_accuracies)
        
        # 计算各项损失
        focal_loss = self.focal_loss(logits, labels)
        aux_focal_loss = self.focal_loss(aux_logits, labels)
        hard_mining_loss = self.hard_mining_loss(logits, labels)
        triangular_loss = self.triangular_loss(processed_features, labels)
        
        # 动态加权
        weights = F.softmax(self.loss_weights, dim=0)
        
        total_loss = (weights[0] * focal_loss + 
                     weights[1] * aux_focal_loss + 
                     weights[2] * hard_mining_loss + 
                     weights[3] * triangular_loss)
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return focal_loss
            
        return total_loss 