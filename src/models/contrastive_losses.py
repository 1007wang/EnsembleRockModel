import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """监督对比学习损失函数"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: 形状为 [batch_size, n_views, ...] 的特征向量
            labels: 形状为 [batch_size] 的标签
        Returns:
            对比学习损失
        """
        device = features.device
        
        # 确保输入维度正确
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        
        # 确保标签维度正确
        if labels.shape[0] != batch_size:
            raise ValueError(f'特征数量 ({batch_size}) 与标签数量 ({labels.shape[0]}) 不匹配')
            
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 计算特征之间的相似度
        features = F.normalize(features, dim=2)
        features = features.view(batch_size, -1)  # 展平特征
        similarity_matrix = torch.matmul(features, features.T)
        
        # 对角线上的元素是自身的相似度，应该排除
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        mask = mask * logits_mask
        
        # 计算正样本对的损失
        exp_similarity = torch.exp(similarity_matrix / self.temperature)
        
        # 计算每个样本的正样本对数量
        num_positives_per_row = mask.sum(1)
        
        # 避免除零
        denominator = exp_similarity.sum(1) - exp_similarity.diagonal()
        log_prob = torch.log(exp_similarity + 1e-7) - torch.log(denominator.view(-1, 1) + 1e-7)
        
        # 只考虑正样本对的log probability
        mean_log_prob_pos = (mask * log_prob).sum(1) / (num_positives_per_row + 1e-7)
        
        # 缩放损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss

class DomainAdversarialLoss(nn.Module):
    """领域对抗损失"""
    def __init__(self, input_dim, hidden_dim=1024):
        super(DomainAdversarialLoss, self).__init__()
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features, alpha=1.0):
        """
        Args:
            features: 特征向量
            alpha: 梯度反转层的系数
        Returns:
            domain_loss: 领域判别损失
        """
        # 梯度反转层
        reverse_features = GradientReversal.apply(features, alpha)
        domain_pred = self.domain_classifier(reverse_features)
        
        # 创建领域标签（0表示源域，1表示目标域）
        batch_size = features.size(0)
        domain_labels = torch.cat([
            torch.zeros(batch_size // 2),
            torch.ones(batch_size - batch_size // 2)
        ]).to(features.device)
        
        domain_loss = F.binary_cross_entropy_with_logits(
            domain_pred.squeeze(), domain_labels.float()
        )
        
        return domain_loss

class GradientReversal(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class AdaptiveContrastiveLoss(nn.Module):
    """自适应对比学习损失"""
    def __init__(self, temperature=0.07, memory_bank_size=4096):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.memory_bank_size = memory_bank_size
        self.register_buffer("memory_bank", None)
        self.register_buffer("memory_labels", None)
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _update_memory_bank(self, features, labels):
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        
        # 初始化记忆库
        if self.memory_bank is None:
            self.memory_bank = torch.zeros(
                (self.memory_bank_size, feature_dim),
                device=features.device
            )
            self.memory_labels = torch.zeros(
                self.memory_bank_size,
                device=features.device,
                dtype=labels.dtype
            )
        
        # 更新记忆库
        ptr = int(self.memory_ptr)
        # 计算可以更新的样本数量
        update_size = min(batch_size, self.memory_bank_size - ptr)
        if update_size > 0:
            self.memory_bank[ptr:ptr + update_size] = features[:update_size]
            self.memory_labels[ptr:ptr + update_size] = labels[:update_size]
            self.memory_ptr[0] = (ptr + update_size) % self.memory_bank_size
        
    def forward(self, features, labels, update_memory=True):
        """
        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]
            update_memory: 是否更新记忆库
        """
        if update_memory:
            self._update_memory_bank(features.detach(), labels)
        
        # 使用当前batch的特征计算对比损失
        return SupConLoss(temperature=self.temperature)(
            features.unsqueeze(1),  # [batch_size, 1, feature_dim]
            labels
        )

class CombinedAdaptiveContrastiveLoss(nn.Module):
    """组合自适应对比损失"""
    def __init__(self, feature_dim, num_classes, 
                 temperature=0.07, memory_bank_size=4096):
        super(CombinedAdaptiveContrastiveLoss, self).__init__()
        
        self.contrastive = AdaptiveContrastiveLoss(
            temperature=temperature,
            memory_bank_size=memory_bank_size
        )
        self.domain_adversarial = DomainAdversarialLoss(
            input_dim=feature_dim
        )
        
    def forward(self, features, labels, domain_features, alpha=1.0):
        """
        Args:
            features: 分类特征
            labels: 类别标签
            domain_features: 领域特征
            alpha: 领域自适应的权重
        """
        contrastive_loss = self.contrastive(features, labels)
        domain_loss = self.domain_adversarial(domain_features, alpha)
        
        return contrastive_loss + alpha * domain_loss 