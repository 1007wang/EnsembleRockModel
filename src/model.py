import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import json
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

class FocalLossWithLabelSmoothing(nn.Module):
    """结合标签平滑的Focal Loss"""
    def __init__(self, num_classes, gamma=2.0, smoothing=0.1):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        
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
        loss = -torch.log(pt) * focal_weight
        
        return loss.mean()

class RockDataset(Dataset):
    """岩石数据集加载器"""
    def __init__(self, data_dir, transform=None, mode='train', train_ratio=0.8):
        """
        初始化数据集
        
        Args:
            data_dir (str): 数据目录路径
            transform: 数据转换操作
            mode (str): 'train' 或 'val'
            train_ratio (float): 训练集比例，默认0.8
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.train_ratio = train_ratio
        
        # 加载类别映射
        with open(os.path.join(data_dir, 'class_mapping.json'), 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
            
        # 收集所有图片路径和标签
        self.samples = []
        self._collect_samples()
        
        # 划分数据集
        self._split_dataset()
        
        # 预加载图片路径到内存
        self.image_paths = [os.path.join(self.data_dir, sample['path']) for sample in self.samples]
        self.labels = [sample['label'] for sample in self.samples]
        
    def _collect_samples(self):
        """收集所有图片样本"""
        for class_path, class_idx in self.class_mapping.items():
            full_path = os.path.join(self.data_dir, class_path)
            if os.path.exists(full_path):
                # 对文件名排序以确保顺序一致性（重要：保证数据集划分可复现）
                for img_name in sorted(os.listdir(full_path)):
                    if img_name.lower().endswith(('.jpg', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(class_path, img_name),
                            'label': class_idx
                        })
                        
    def _split_dataset(self):
        """划分数据集，处理类别不平衡"""
        samples_by_class = defaultdict(list)
        for sample in self.samples:
            samples_by_class[sample['label']].append(sample)
        
        train_samples = []
        val_samples = []
        
        # 使用固定的随机种子
        np.random.seed(42)
        
        for label, samples in samples_by_class.items():
            n_samples = len(samples)
            # 打乱样本顺序
            indices = np.random.permutation(n_samples)
            samples = [samples[i] for i in indices]
            
            # 根据样本数量动态调整验证集比例
            if n_samples < 100:  # 样本数少于100的类别
                val_size = min(10, n_samples // 3)  # 取10个或1/3，取较小值
            elif n_samples < 300:  # 样本数在100-300之间的类别
                val_size = min(30, n_samples // 4)  # 取30个或1/4，取较小值
            elif n_samples < 800:  # 样本数在300-800之间的类别
                val_size = int(n_samples * 0.2)  # 取20%
            else:  # 样本数大于800的类别
                val_size = int(n_samples * 0.15)  # 取15%
            
            n_train = n_samples - val_size
            
            if self.mode == 'train':
                train_samples.extend(samples[:n_train])
            else:
                val_samples.extend(samples[n_train:])
        
        self.samples = train_samples if self.mode == 'train' else val_samples
        
        # 打印数据集统计信息
        if self.mode == 'train':
            print(f"\n训练集样本统计:")
            print(f"总样本数: {len(self.samples)}")
            class_counts = {}
            for sample in self.samples:
                label = sample['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            print(f"类别数量: {len(class_counts)}")
        else:
            print(f"\n验证集样本统计:")
            print(f"总样本数: {len(self.samples)}")
            class_counts = {}
            for sample in self.samples:
                label = sample['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            print(f"类别数量: {len(class_counts)}")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 使用 PIL 加载图片
            with Image.open(img_path) as img:
                # 确保图片是RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 确保图片尺寸足够大
                if img.size[0] < 299 or img.size[1] < 299:
                    img = img.resize((320, 320), Image.BICUBIC)
                
                # 应用转换
                if self.transform:
                    try:
                        img = self.transform(img)
                    except Exception as e:
                        print(f"Error transforming image {img_path}: {str(e)}")
                        # 如果转换失败，使用基本转换
                        img = transforms.Compose([
                            transforms.Resize((299, 299)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                        ])(img)
                
                return img, label
                
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回一个有效的默认张量
            default_tensor = torch.zeros((3, 299, 299))
            default_tensor.normal_(0, 0.1)
            default_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(default_tensor)
            return default_tensor, label

def get_weighted_sampler(dataset):
    """改进的加权采样器"""
    targets = [sample['label'] for sample in dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets))
    
    # 使用平方根重采样策略
    weights = 1.0 / torch.sqrt(class_counts.float())
    weights = weights / weights.sum()
    
    # 为每个样本分配权重
    sample_weights = weights[targets]
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def get_transforms(mode='train'):
    """优化的数据增强策略"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(p=0.3),  # 降低翻转概率
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(10),  # 减小旋转角度
            transforms.ColorJitter(
                brightness=0.15,  # 减小颜色增强强度
                contrast=0.15,
                saturation=0.15,
                hue=0.05
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1)  # 降低擦除概率
        ])
    else:
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.CenterCrop(299),  # 使用中心裁剪替代简单缩放
            transforms.ToTensor(),
            normalize
        ])

def create_data_loaders(data_dir, batch_size=32, num_workers=8, train_ratio=0.8):
    """优化后的数据加载器"""
    train_dataset = RockDataset(
        data_dir=data_dir,
        transform=get_transforms('train'),
        mode='train',
        train_ratio=train_ratio
    )
    
    val_dataset = RockDataset(
        data_dir=data_dir,
        transform=get_transforms('val'),
        mode='val',
        train_ratio=train_ratio
    )
    
    # 使用加权采样器
    train_sampler = get_weighted_sampler(train_dataset)
    
    # 优化数据加载参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=min(16, os.cpu_count() * 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(16, os.cpu_count() * 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        drop_last=False
    )
    
    return train_loader, val_loader