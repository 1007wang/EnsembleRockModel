import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised contrastive learning loss function"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: Feature vectors with shape [batch_size, n_views, ...]
            labels: Labels with shape [batch_size]
        Returns:
            Contrastive learning loss
        """
        device = features.device
        
        # Ensure input dimensions are correct
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        
        # Ensure label dimensions are correct
        if labels.shape[0] != batch_size:
            raise ValueError(f'Number of features ({batch_size}) does not match number of labels ({labels.shape[0]})')
            
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Calculate similarity between features
        features = F.normalize(features, dim=2)
        features = features.view(batch_size, -1)  # Flatten features
        similarity_matrix = torch.matmul(features, features.T)
        
        # Diagonal elements are self-similarity, should be excluded
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        mask = mask * logits_mask
        
        # Calculate loss for positive pairs
        exp_similarity = torch.exp(similarity_matrix / self.temperature)
        
        # Calculate number of positive pairs per sample
        num_positives_per_row = mask.sum(1)
        
        # Avoid division by zero
        denominator = exp_similarity.sum(1) - exp_similarity.diagonal()
        log_prob = torch.log(exp_similarity + 1e-7) - torch.log(denominator.view(-1, 1) + 1e-7)
        
        # Only consider log probability of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (num_positives_per_row + 1e-7)
        
        # Scale loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss

class DomainAdversarialLoss(nn.Module):
    """Domain adversarial loss"""
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
            features: Feature vectors
            alpha: Coefficient for gradient reversal layer
        Returns:
            domain_loss: Domain discrimination loss
        """
        # Gradient reversal layer
        reverse_features = GradientReversal.apply(features, alpha)
        domain_pred = self.domain_classifier(reverse_features)
        
        # Create domain labels (0 for source domain, 1 for target domain)
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
    """Gradient reversal layer"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class AdaptiveContrastiveLoss(nn.Module):
    """Adaptive contrastive learning loss"""
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
        
        # Initialize memory bank
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
        
        # Update memory bank
        ptr = int(self.memory_ptr)
        # Calculate number of samples that can be updated
        update_size = min(batch_size, self.memory_bank_size - ptr)
        if update_size > 0:
            self.memory_bank[ptr:ptr + update_size] = features[:update_size]
            self.memory_labels[ptr:ptr + update_size] = labels[:update_size]
            self.memory_ptr[0] = (ptr + update_size) % self.memory_bank_size
        
    def forward(self, features, labels, update_memory=True):
        """
        Args:
            features: Feature vectors [batch_size, feature_dim]
            labels: Labels [batch_size]
            update_memory: Whether to update memory bank
        """
        if update_memory:
            self._update_memory_bank(features.detach(), labels)
        
        # Calculate contrastive loss using current batch features
        return SupConLoss(temperature=self.temperature)(
            features.unsqueeze(1),  # [batch_size, 1, feature_dim]
            labels
        )

class CombinedAdaptiveContrastiveLoss(nn.Module):
    """Combined adaptive contrastive loss"""
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
            features: Classification features
            labels: Class labels
            domain_features: Domain features
            alpha: Weight for domain adaptation
        """
        contrastive_loss = self.contrastive(features, labels)
        domain_loss = self.domain_adversarial(domain_features, alpha)
        
        return contrastive_loss + alpha * domain_loss 