import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging
import argparse
from models.ensemble_model import EnsembleModel
from models.backbone_models import create_model, create_model_without_attention
from model import create_data_loaders
import matplotlib.pyplot as plt
import seaborn as sns
import math

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def setup_matplotlib():
    """配置matplotlib支持中文显示"""
    import matplotlib
    # 强制使用Agg后端，避免GUI相关问题
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # 尝试设置中文字体
    try:
        # 尝试使用系统中的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 检查是否成功设置了中文字体
        fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = [f for f in fonts if '黑体' in f or '雅黑' in f or 'SimSun' in f or 'SimHei' in f]
        
        if not chinese_fonts:
            # 如果没有找到中文字体，使用英文
            print("警告: 未找到中文字体，将使用英文显示")
            use_chinese = False
        else:
            print(f"找到中文字体: {chinese_fonts[0]}")
            use_chinese = True
    except Exception as e:
        print(f"设置中文字体时出错: {str(e)}，将使用英文显示")
        use_chinese = False
    
    return use_chinese

def setup_logger(log_dir):
    """设置日志"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'ensemble_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger('ensemble_training')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, val_acc, epoch):
        """
        检查是否应该早停
        
        参数:
            val_loss: 验证损失
            val_acc: 验证准确率
            epoch: 当前轮次
        
        返回:
            bool: 是否应该早停
        """
        # 第一次调用时初始化best_acc
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_epoch = epoch
            return False
        # 如果验证准确率提高，更新最佳准确率和轮次
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
        # 如果连续patience轮验证准确率没有提高，触发早停
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class LossMonitor:
    def __init__(self, threshold=5):
        self.threshold = threshold
        self.consecutive_invalid = 0
        self.last_valid_state = None
        
    def check_loss(self, loss, model, optimizer):
        if torch.isnan(loss) or torch.isinf(loss):
            self.consecutive_invalid += 1
            if self.consecutive_invalid >= self.threshold and self.last_valid_state is not None:
                print("检测到连续无效损失，恢复到上一个有效状态")
                model.load_state_dict(self.last_valid_state['model'])
                optimizer.load_state_dict(self.last_valid_state['optimizer'])
                return False
        else:
            self.consecutive_invalid = 0
            self.last_valid_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
        return True

def train_epoch(model, train_loader, optimizer, scaler, device, epoch, total_epochs, gradient_clip, class_accuracies=None, accumulation_steps=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 计算当前的域自适应权重
    alpha = 2.0 / (1.0 + np.exp(-10 * epoch / total_epochs)) - 1.0
    
    # 计算当前的对比学习权重
    contrast_weight = min(0.1 * (epoch / 10), 0.5)  # 逐渐增加对比学习的权重
    
    with tqdm(train_loader, desc='Training', ncols=100) as pbar:
        for idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(device_type='cuda:1' if torch.cuda.is_available() else 'cpu'):
                # 根据模型类型决定是否传入contrast_weight参数
                if isinstance(model, EnsembleModel):
                    # EnsembleModel支持contrast_weight参数
                    outputs, loss = model(
                        inputs, 
                        labels, 
                        alpha=alpha,
                        class_accuracies=class_accuracies,
                        contrast_weight=contrast_weight
                    )
                else:
                    # 其他模型不支持contrast_weight参数
                    outputs, loss = model(
                        inputs, 
                        labels, 
                        alpha=alpha,
                        class_accuracies=class_accuracies
                    )
                
                # 检查损失值是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss value detected: {loss}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            
            # 反向传播
            scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0:
                # 添加梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # 统计
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss/(idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'alpha': f'{alpha:.3f}'
            })
            
            if idx % 500 == 0:
                torch.cuda.empty_cache()
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, device):
    """验证模型性能"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    print("\n开始验证...")
    with torch.no_grad():
        # 使用tqdm创建进度条，设置leave=True使进度条保留
        for inputs, labels in tqdm(val_loader, desc="验证进度", leave=True, ncols=100):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 根据设备类型确定autocast的device_type
            device_type = 'cuda:1' if device.type == 'cuda:1' else 'cpu'
            
            with autocast(device_type=device_type):
                outputs = model(inputs)
                # 处理outputs为元组的情况，取第一个元素作为logits
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算准确率
    accuracy = np.mean(all_preds == all_labels)
    
    # 计算混淆矩阵
    num_classes = len(np.unique(all_labels))
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        confusion_mat[t, p] += 1
    
    # 计算每个类别的准确率
    class_accuracies = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    class_support = {}
    
    for class_id in range(num_classes):
        # 该类别的样本总数
        class_total = np.sum(all_labels == class_id)
        if class_total > 0:
            # 该类别被正确分类的样本数
            class_correct = confusion_mat[class_id, class_id]
            # 准确率 = 正确分类的样本数 / 该类别的样本总数
            class_accuracies[class_id] = class_correct / class_total
            
            # 计算精确率 (precision) = TP / (TP + FP)
            # TP = 正确预测为该类别的样本数
            # FP = 错误预测为该类别的样本数
            predicted_as_class = np.sum(confusion_mat[:, class_id])
            if predicted_as_class > 0:
                class_precision[class_id] = confusion_mat[class_id, class_id] / predicted_as_class
            else:
                class_precision[class_id] = 0.0
            
            # 计算召回率 (recall) = TP / (TP + FN)
            # TP = 正确预测为该类别的样本数
            # FN = 该类别被错误预测为其他类别的样本数
            class_recall[class_id] = confusion_mat[class_id, class_id] / class_total
            
            # 计算F1分数 = 2 * (precision * recall) / (precision + recall)
            if class_precision[class_id] + class_recall[class_id] > 0:
                class_f1[class_id] = 2 * (class_precision[class_id] * class_recall[class_id]) / (class_precision[class_id] + class_recall[class_id])
            else:
                class_f1[class_id] = 0.0
                
            # 记录支持度（样本数）
            class_support[class_id] = int(class_total)
    
    # 计算宏平均和加权平均指标
    macro_precision = np.mean(list(class_precision.values()))
    macro_recall = np.mean(list(class_recall.values()))
    macro_f1 = np.mean(list(class_f1.values()))
    
    # 计算加权平均，考虑每个类别的样本数
    weights = np.array([class_support[i] for i in range(num_classes)])
    weights = weights / np.sum(weights)
    
    weighted_precision = np.sum([class_precision[i] * weights[i] for i in range(num_classes)])
    weighted_recall = np.sum([class_recall[i] * weights[i] for i in range(num_classes)])
    weighted_f1 = np.sum([class_f1[i] * weights[i] for i in range(num_classes)])
    
    # 返回验证结果
    return {
        'accuracy': accuracy,
        'loss': val_loss / len(val_loader),
        'confusion_matrix': confusion_mat,
        'class_accuracies': class_accuracies,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'class_support': class_support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def generate_low_accuracy_class_heatmaps(model, val_loader, device, class_mapping, accuracy_threshold=0.75, max_per_class=5, max_total=50, save_dir='visualization'):
    """生成低准确率类别的热力图"""
    print("\n开始生成低准确率类别的热力图...")
    
    # 确保模型处于评估模式
    model.eval()
    
    # 创建反向映射（id到名称）
    id_to_name = {v: k for k, v in class_mapping.items()}
    
    # 首先进行一次验证，获取每个类别的准确率
    val_results = validate(model, val_loader, device)
    class_accuracies = val_results['class_accuracies']
    
    # 找出准确率低于阈值的类别
    low_accuracy_classes = {class_id: acc for class_id, acc in class_accuracies.items() if acc < accuracy_threshold}
    
    # 如果没有低于阈值的类别，选择准确率最低的10个类别
    if not low_accuracy_classes:
        print(f"没有找到准确率低于{accuracy_threshold*100:.1f}%的类别，将选择准确率最低的10个类别")
        sorted_accuracies = sorted(class_accuracies.items(), key=lambda x: x[1])
        low_accuracy_classes = {class_id: acc for class_id, acc in sorted_accuracies[:10]}
    
    print(f"找到{len(low_accuracy_classes)}个低准确率类别:")
    for class_id, accuracy in low_accuracy_classes.items():
        class_name = id_to_name[class_id]
        print(f"  - {class_name}: {accuracy*100:.2f}%")
    
    # 收集这些类别的错误预测样本
    incorrect_samples = {class_id: [] for class_id in low_accuracy_classes.keys()}
    
    model.eval()
    print("\n开始收集错误预测样本...")
    with torch.no_grad():
        # 使用tqdm创建进度条，设置leave=True使进度条保留
        for inputs, labels in tqdm(val_loader, desc="收集错误预测样本", leave=True, ncols=100):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # 处理outputs为元组的情况，取第一个元素作为logits
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            _, preds = torch.max(outputs, 1)
            
            # 找出预测错误的样本
            for i, (label, pred) in enumerate(zip(labels, preds)):
                label_item = label.item()
                pred_item = pred.item()
                
                # 如果是低准确率类别且预测错误
                if label_item in low_accuracy_classes and label_item != pred_item:
                    # 只保存指定数量的样本
                    if len(incorrect_samples[label_item]) < max_per_class:
                        incorrect_samples[label_item].append({
                            'image': inputs[i].cpu(),
                            'true_label': label_item,
                            'pred_label': pred_item
                        })
    
    # 确保可视化目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个低准确率类别生成热力图
    total_generated = 0
    
    for class_id, samples in incorrect_samples.items():
        class_name = id_to_name[class_id]
        accuracy = low_accuracy_classes[class_id]
        
        print(f"\n为类别 {class_name} (准确率: {accuracy*100:.2f}%) 生成热力图")
        
        # 限制每个类别的热力图数量
        for i, sample in enumerate(samples):
            # 限制总热力图数量
            if total_generated >= max_total:
                print(f"已达到最大热力图数量限制 ({max_total})")
                break
                
            try:
                # 生成Grad-CAM
                with torch.enable_grad():
                    cam = model.grad_cam.generate_cam(sample['image'].unsqueeze(0).to(device), sample['true_label'])
                
                # 处理图像
                image = sample['image'].numpy().transpose(1, 2, 0)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                overlayed_image = model.grad_cam.overlay_cam(image, cam)
                
                # 构建安全的文件名
                safe_class_name = get_class_basename(class_name).replace('\\', '_').replace('/', '_')
                save_path = os.path.join(save_dir, f'grad_cam_{safe_class_name}_acc{accuracy*100:.1f}_sample{i+1}.png')
                
                # 获取预测类别名称
                pred_class_name = get_class_basename(id_to_name[sample['pred_label']])
                
                # 保存热力图
                plt.figure(figsize=(10, 10))
                plt.imshow(overlayed_image)
                
                # 获取真实类别的最后部分
                true_class_display = get_class_basename(class_name)
                
                plt.title(f'真实类别: {true_class_display}\n预测类别: {pred_class_name}\n类别准确率: {accuracy*100:.2f}%')
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()  # 立即关闭图像释放内存
                
                print(f"已保存热力图: {save_path}")
                total_generated += 1
                
            except Exception as e:
                print(f"生成类别 {class_name} 的热力图时发生错误: {str(e)}")
                continue
            
            # 每处理5张图像清理一次内存
            if total_generated % 5 == 0:
                torch.cuda.empty_cache()
    
    print(f"\n热力图生成完成，共生成 {total_generated} 张热力图")

def visualize_confusion_matrix(confusion_matrix, class_names, epoch, save_dir='visualization', use_chinese=False):
    """可视化混淆矩阵"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建混淆矩阵图
    plt.figure(figsize=(20, 16))
    
    # 获取类别的基本名称
    display_names = [get_class_basename(name) for name in class_names]
    
    # 创建热力图
    sns.heatmap(
        confusion_matrix, 
        annot=False,  # 不显示数值，太多类别会重叠
        cmap='Blues',
        fmt='d', 
        xticklabels=display_names,
        yticklabels=display_names
    )
    
    # 根据是否支持中文选择标题
    if use_chinese:
        plt.title(f'第{epoch+1}轮 混淆矩阵', fontsize=16)
        plt.xlabel('预测类别', fontsize=14)
        plt.ylabel('真实类别', fontsize=14)
    else:
        plt.title(f'Confusion Matrix - Epoch {epoch+1}', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('True Class', fontsize=14)
    
    # 调整标签大小和旋转角度
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png'), dpi=300)
    plt.close()

def analyze_performance(val_results, class_mapping, epoch, save_dir='visualization', use_chinese=False):
    """分析模型性能，生成性能报告"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建反向映射（id到名称）
    id_to_name = {v: get_class_basename(k) for k, v in class_mapping.items()}
    
    # 获取每个类别的性能指标
    class_metrics = {}
    for class_id, accuracy in val_results['class_accuracies'].items():
        class_name = id_to_name[class_id]
        class_metrics[class_name] = {
            'accuracy': accuracy,
            'precision': val_results['class_precision'][class_id],
            'recall': val_results['class_recall'][class_id],
            'f1': val_results['class_f1'][class_id],
            'support': val_results['class_support'][class_id]
        }
    
    # 按F1分数排序
    sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # 创建性能报告
    report_file = os.path.join(save_dir, f'performance_report_epoch_{epoch+1}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        # 写入总体性能
        if use_chinese:
            f.write(f"第{epoch+1}轮模型性能报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"总体准确率: {val_results['accuracy']*100:.2f}%\n")
            f.write(f"宏平均F1: {val_results['macro_f1']:.4f}\n")
            f.write(f"加权平均F1: {val_results['weighted_f1']:.4f}\n\n")
            f.write("各类别性能指标:\n")
            f.write("-"*50 + "\n")
        else:
            f.write(f"Model Performance Report - Epoch {epoch+1}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall Accuracy: {val_results['accuracy']*100:.2f}%\n")
            f.write(f"Macro F1: {val_results['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {val_results['weighted_f1']:.4f}\n\n")
            f.write("Class-wise Metrics:\n")
            f.write("-"*50 + "\n")
        
        # 写入每个类别的性能
        for class_name, metrics in sorted_classes:
            if use_chinese:
                f.write(f"类别: {class_name}\n")
                f.write(f"  准确率: {metrics['accuracy']*100:.2f}%\n")
                f.write(f"  精确率: {metrics['precision']:.4f}\n")
                f.write(f"  召回率: {metrics['recall']:.4f}\n")
                f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                f.write(f"  样本数: {metrics['support']}\n")
            else:
                f.write(f"Class: {class_name}\n")
                f.write(f"  Accuracy: {metrics['accuracy']*100:.2f}%\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
            f.write("-"*50 + "\n")
        
        # 写入性能最好和最差的类别
        best_classes = sorted_classes[:5]
        worst_classes = sorted_classes[-5:]
        
        if use_chinese:
            f.write("\n性能最好的5个类别:\n")
        else:
            f.write("\nTop 5 Best Performing Classes:\n")
        f.write("-"*50 + "\n")
        for i, (class_name, metrics) in enumerate(best_classes):
            if use_chinese:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, 准确率: {metrics['accuracy']*100:.2f}%\n")
            else:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%\n")
        
        if use_chinese:
            f.write("\n性能最差的5个类别:\n")
        else:
            f.write("\nTop 5 Worst Performing Classes:\n")
        f.write("-"*50 + "\n")
        for i, (class_name, metrics) in enumerate(reversed(worst_classes)):
            if use_chinese:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, 准确率: {metrics['accuracy']*100:.2f}%\n")
            else:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%\n")
    
    # 可视化性能最差的类别
    plt.figure(figsize=(12, 8))
    worst_class_names = [name for name, _ in reversed(worst_classes)]
    worst_class_f1 = [metrics['f1'] for _, metrics in reversed(worst_classes)]
    
    plt.bar(range(len(worst_class_names)), worst_class_f1)
    plt.xticks(range(len(worst_class_names)), worst_class_names, rotation=45, ha='right')
    
    if use_chinese:
        plt.title(f'第{epoch+1}轮性能最差的5个类别的F1分数')
        plt.xlabel('类别')
        plt.ylabel('F1分数')
    else:
        plt.title(f'F1 Scores of 5 Worst Performing Classes - Epoch {epoch+1}')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'worst_classes_epoch_{epoch+1}.png'))
    plt.close()
    
    # 可视化性能最好的类别
    plt.figure(figsize=(12, 8))
    best_class_names = [name for name, _ in best_classes]
    best_class_f1 = [metrics['f1'] for _, metrics in best_classes]
    
    plt.bar(range(len(best_class_names)), best_class_f1)
    plt.xticks(range(len(best_class_names)), best_class_names, rotation=45, ha='right')
    
    if use_chinese:
        plt.title(f'第{epoch+1}轮性能最好的5个类别的F1分数')
        plt.xlabel('类别')
        plt.ylabel('F1分数')
    else:
        plt.title(f'F1 Scores of 5 Best Performing Classes - Epoch {epoch+1}')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'best_classes_epoch_{epoch+1}.png'))
    plt.close()
    
    return report_file

def plot_learning_curves(train_history, save_dir='visualization', use_chinese=False):
    """Plot learning curves with unified English style, supporting dynamic learning rate stage visualization"""
    # Apply unified plotting style (consistent with english_detailed_line_charts_generator.py)
    plt.rcParams.update({
        'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 12,
        'axes.unicode_minus': False,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white'
    })
    
    # Prepare data
    epochs = range(1, len(train_history['train_loss']) + 1)
    
    # Create canvas and subplots - increased size for better visibility
    fig, axes = plt.subplots(3, 2, figsize=(22, 24))
    fig.suptitle('Training Process Analysis - Learning Curves', fontsize=20, fontweight='bold', y=0.995)
    
    # Define unified color scheme
    train_color = '#FF8A8A'  # Red for training
    val_color = '#66B3FF'    # Blue for validation
    highlight_color = '#90EE90'  # Green for highlights
    
    # 1. Loss curve (decreasing curve - legend at top right)
    ax = axes[0, 0]
    ax.plot(epochs, train_history['train_loss'], color=train_color, linewidth=2.5, 
            label='Training Loss', alpha=0.88)
    ax.plot(epochs, train_history['val_loss'], color=val_color, linewidth=2.5,
            label='Validation Loss', alpha=0.88)
    
    # Find minimum validation loss
    min_val_loss_epoch = train_history['val_loss'].index(min(train_history['val_loss'])) + 1
    ax.axvline(x=min_val_loss_epoch, color=highlight_color, linestyle='--', linewidth=1.5,
               label=f'Min Val Loss (Epoch {min_val_loss_epoch})', alpha=0.7)
    
    # Add panel label
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
           fontsize=18, fontweight='bold', verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Training Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='gray')  # Top right for decreasing curves
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 2. Accuracy curve (increasing curve - legend at bottom right)
    ax = axes[0, 1]
    ax.plot(epochs, [acc * 100 for acc in train_history['train_acc']], color=train_color, 
            linewidth=2.5, label='Training Accuracy', alpha=0.88)
    ax.plot(epochs, [acc * 100 for acc in train_history['val_acc']], color=val_color,
            linewidth=2.5, label='Validation Accuracy', alpha=0.88)
    
    # Find maximum validation accuracy
    max_val_acc_epoch = train_history['val_acc'].index(max(train_history['val_acc'])) + 1
    ax.axvline(x=max_val_acc_epoch, color=highlight_color, linestyle='--', linewidth=1.5,
               label=f'Max Val Acc (Epoch {max_val_acc_epoch})', alpha=0.7)
    
    # Add panel label
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
           fontsize=18, fontweight='bold', verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Training Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='gray')  # Bottom right for increasing curves
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 3. Learning rate curve - Dynamic stage boundaries
    ax = axes[1, 0]
    
    # Plot learning rate curve
    ax.plot(epochs, train_history['learning_rates'], color='#FFD699', linewidth=2.5,
            label='Learning Rate', alpha=0.88)
    
    # Get dynamic stage boundaries
    stage_boundaries = train_history.get('stage_boundaries', {})
    
    # Add stage divisions and color regions based on actual recorded boundaries
    colors = {'warmup': '#FF8A8A', 'aggressive': '#66B3FF', 'refinement': '#C77EFF', 'fine_tuning': '#FFD699'}
    labels = {'warmup': 'Warmup', 'aggressive': 'Aggressive', 'refinement': 'Refinement', 'fine_tuning': 'Fine-tuning'}
    
    # Check if stage information is recorded
    if 'lr_stages' in train_history and len(train_history['lr_stages']) > 0:
        # Determine epoch ranges for each stage
        stage_ranges = {}
        current_stage = train_history['lr_stages'][0]
        start_epoch = 1
        
        for i, stage in enumerate(train_history['lr_stages']):
            if stage != current_stage or i == len(train_history['lr_stages']) - 1:
                # Stage change or last epoch
                end_epoch = i if stage != current_stage else i + 1
                stage_ranges[current_stage] = (start_epoch, end_epoch)
                current_stage = stage
                start_epoch = i + 1
        
        # Ensure last stage is properly recorded
        if current_stage not in stage_ranges:
            stage_ranges[current_stage] = (start_epoch, len(train_history['lr_stages']))
        
        # Draw each stage
        for stage, (start, end) in stage_ranges.items():
            if stage in colors:
                # Add stage color region
                ax.axvspan(start, end, alpha=0.1, color=colors[stage])
                
                # Add stage label
                mid_point = (start + end) / 2
                if mid_point > 0 and mid_point < len(epochs):
                    ax.text(mid_point, max(train_history['learning_rates']) * 0.9, 
                           labels.get(stage, stage), 
                           horizontalalignment='center', color=colors[stage],
                           fontsize=10, fontweight='bold')
        
        # Add stage division lines
        for stage, boundary in stage_boundaries.items():
            if boundary > 0 and stage.endswith('_end'):
                stage_name = stage.replace('_end', '')
                ax.axvline(x=boundary, color=colors.get(stage_name, 'k'), linestyle='--', 
                          linewidth=1.5, alpha=0.6)
    
    # Add panel label
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes,
           fontsize=18, fontweight='bold', verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Training Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('Learning Rate Schedule (Dynamic Stages)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 4. F1 score curve (increasing curve - legend at bottom right)
    ax = axes[1, 1]
    ax.plot(epochs, train_history['macro_f1'], color='#FF8A8A', linewidth=2.5,
            label='Macro F1', alpha=0.88)
    ax.plot(epochs, train_history['weighted_f1'], color='#66B3FF', linewidth=2.5,
            label='Weighted F1', alpha=0.88)
    
    # Find maximum F1 scores
    max_macro_f1_epoch = train_history['macro_f1'].index(max(train_history['macro_f1'])) + 1
    max_weighted_f1_epoch = train_history['weighted_f1'].index(max(train_history['weighted_f1'])) + 1
    
    ax.axvline(x=max_macro_f1_epoch, color='#90EE90', linestyle='--', linewidth=1.5,
               label=f'Max Macro F1 (Epoch {max_macro_f1_epoch})', alpha=0.7)
    ax.axvline(x=max_weighted_f1_epoch, color='#C77EFF', linestyle='--', linewidth=1.5,
               label=f'Max Weighted F1 (Epoch {max_weighted_f1_epoch})', alpha=0.7)
    
    # Add panel label
    ax.text(0.02, 0.98, '(d)', transform=ax.transAxes,
           fontsize=18, fontweight='bold', verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Training Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('F1 Scores', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='gray')  # Bottom right for increasing curves
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 5. Precision & Recall curve (increasing curve - legend at bottom right)
    ax = axes[2, 0]
    
    # Calculate average precision and recall
    avg_precision = []
    avg_recall = []
    
    for epoch_idx in range(len(epochs)):
        epoch_precisions = []
        epoch_recalls = []
        
        for class_name in train_history['class_precision']:
            if epoch_idx < len(train_history['class_precision'][class_name]):
                epoch_precisions.append(train_history['class_precision'][class_name][epoch_idx])
                
        for class_name in train_history['class_recall']:
            if epoch_idx < len(train_history['class_recall'][class_name]):
                epoch_recalls.append(train_history['class_recall'][class_name][epoch_idx])
                
        avg_precision.append(sum(epoch_precisions) / len(epoch_precisions) if epoch_precisions else 0)
        avg_recall.append(sum(epoch_recalls) / len(epoch_recalls) if epoch_recalls else 0)
    
    ax.plot(epochs, avg_precision, color='#FF8A8A', linewidth=2.5,
            label='Average Precision', alpha=0.88)
    ax.plot(epochs, avg_recall, color='#66B3FF', linewidth=2.5,
            label='Average Recall', alpha=0.88)
    
    # Add panel label
    ax.text(0.02, 0.98, '(e)', transform=ax.transAxes,
           fontsize=18, fontweight='bold', verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Training Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Average Precision and Recall', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='gray')  # Bottom right for increasing curves
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 6. Combined metric: Validation accuracy vs F1 relationship
    ax = axes[2, 1]
    
    # Create combined scatter plot
    sc = ax.scatter(
        [acc * 100 for acc in train_history['val_acc']], 
        train_history['weighted_f1'],
        c=list(epochs),  # Use epochs as color
        cmap='viridis',
        marker='o',
        s=50,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Epoch', fontsize=11, fontweight='bold')
    
    # Mark best point
    best_idx = train_history['weighted_f1'].index(max(train_history['weighted_f1']))
    best_acc = train_history['val_acc'][best_idx] * 100
    best_f1 = train_history['weighted_f1'][best_idx]
    
    ax.scatter([best_acc], [best_f1], c='#EF5350', s=200, marker='*', 
               label=f'Best Performance (Epoch {best_idx+1})',
               edgecolors='white', linewidths=1.5, zorder=10)
    
    # Add panel label
    ax.text(0.02, 0.98, '(f)', transform=ax.transAxes,
           fontsize=18, fontweight='bold', verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Weighted F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('Validation Accuracy vs Weighted F1 Score', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.995], h_pad=3.0, w_pad=3.0)
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    
    # Close figure to release memory
    plt.close()

def analyze_confusion_pairs(confusion_matrix, class_mapping, top_n=20, save_dir='visualization', use_chinese=False):
    """分析混淆矩阵，找出最容易混淆的类别对"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建反向映射（id到名称）
    id_to_name = {v: get_class_basename(k) for k, v in class_mapping.items()}
    
    # 找出非对角线元素中值最大的元素（最容易混淆的类别对）
    num_classes = confusion_matrix.shape[0]
    confusion_pairs = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > 0:
                # 计算混淆率（被错误分类的样本数 / 该类别的总样本数）
                true_class_total = confusion_matrix[i].sum()
                error_rate = confusion_matrix[i, j] / true_class_total if true_class_total > 0 else 0
                
                confusion_pairs.append({
                    'true_class_id': i,
                    'pred_class_id': j,
                    'true_class_name': id_to_name[i],
                    'pred_class_name': id_to_name[j],
                    'count': int(confusion_matrix[i, j]),
                    'error_rate': error_rate,
                    'true_class_total': int(true_class_total)
                })
    
    # 按混淆次数排序
    confusion_pairs_by_count = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)[:top_n]
    
    # 按错误率排序
    confusion_pairs_by_rate = sorted(confusion_pairs, key=lambda x: x['error_rate'], reverse=True)[:top_n]
    
    # 生成报告
    report_file = os.path.join(save_dir, 'confusion_pairs_analysis.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        if use_chinese:
            f.write("混淆类别对分析\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"按混淆次数排序的前{top_n}个类别对:\n")
            f.write("-"*50 + "\n")
            for i, pair in enumerate(confusion_pairs_by_count):
                f.write(f"{i+1}. 真实类别: {pair['true_class_name']} -> 预测类别: {pair['pred_class_name']}\n")
                f.write(f"   混淆次数: {pair['count']}, 错误率: {pair['error_rate']*100:.2f}%, 真实类别总样本数: {pair['true_class_total']}\n")
            
            f.write("\n按错误率排序的前{top_n}个类别对:\n")
            f.write("-"*50 + "\n")
            for i, pair in enumerate(confusion_pairs_by_rate):
                f.write(f"{i+1}. 真实类别: {pair['true_class_name']} -> 预测类别: {pair['pred_class_name']}\n")
                f.write(f"   混淆次数: {pair['count']}, 错误率: {pair['error_rate']*100:.2f}%, 真实类别总样本数: {pair['true_class_total']}\n")
        else:
            f.write("Confusion Pairs Analysis\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Top {top_n} Confusion Pairs by Count:\n")
            f.write("-"*50 + "\n")
            for i, pair in enumerate(confusion_pairs_by_count):
                f.write(f"{i+1}. True Class: {pair['true_class_name']} -> Predicted Class: {pair['pred_class_name']}\n")
                f.write(f"   Confusion Count: {pair['count']}, Error Rate: {pair['error_rate']*100:.2f}%, True Class Total: {pair['true_class_total']}\n")
            
            f.write(f"\nTop {top_n} Confusion Pairs by Error Rate:\n")
            f.write("-"*50 + "\n")
            for i, pair in enumerate(confusion_pairs_by_rate):
                f.write(f"{i+1}. True Class: {pair['true_class_name']} -> Predicted Class: {pair['pred_class_name']}\n")
                f.write(f"   Confusion Count: {pair['count']}, Error Rate: {pair['error_rate']*100:.2f}%, True Class Total: {pair['true_class_total']}\n")
    
    # 可视化混淆对
    plt.figure(figsize=(14, 8))
    
    # 绘制前10个混淆对的柱状图
    top_10_pairs = confusion_pairs_by_count[:10]
    pair_labels = [f"{p['true_class_name']}->{p['pred_class_name']}" for p in top_10_pairs]
    pair_counts = [p['count'] for p in top_10_pairs]
    
    plt.bar(range(len(pair_labels)), pair_counts)
    plt.xticks(range(len(pair_labels)), pair_labels, rotation=45, ha='right')
    
    if use_chinese:
        plt.title('最容易混淆的10个类别对')
        plt.xlabel('类别对')
        plt.ylabel('混淆次数')
    else:
        plt.title('Top 10 Most Confused Class Pairs')
        plt.xlabel('Class Pairs')
        plt.ylabel('Confusion Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_pairs.png'))
    plt.close()
    
    return report_file

def get_class_basename(class_name):
    """从类别路径中提取基本名称，处理不同的路径分隔符"""
    if '\\' in class_name:
        return class_name.split('\\')[-1]
    elif '/' in class_name:
        return class_name.split('/')[-1]
    else:
        return class_name

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='岩石分类模型训练与评估')
    
    # 配置文件支持
    parser.add_argument('--config', type=str, default=None, 
                        help='JSON配置文件路径，会覆盖命令行参数')
    
    # 模型选择参数
    parser.add_argument('--model_type', type=str, default='ensemble',
                       choices=['ensemble', 'resnet50', 'resnet50_optimized', 'efficientnet_b4', 'inceptionv3'],
                       help='选择要训练的模型类型')
    parser.add_argument('--no_attention', action='store_true',
                       help='设置此参数将禁用注意力机制')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler_type', type=str, default='multistage',
                       choices=['multistage', 'cosine'],
                       help='选择学习率调度器类型，multistage为多阶段，cosine为单一余弦')
    parser.add_argument('--cosine_T_max', type=int, default=None,
                       help='余弦调度器的T_max参数，默认等于总轮次')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6,
                       help='余弦调度器的最小学习率')
    parser.add_argument('--cosine_warmup_epochs', type=int, default=10,
                       help='余弦调度器的预热轮次')
    
    # 训练参数
    parser.add_argument('--data_dir', type=str, default='processed_data', help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='梯度累积步数')
    
    # 评估参数
    parser.add_argument('--accuracy_threshold', type=float, default=0.8, 
                        help='热力图生成的准确率阈值，低于此阈值的类别将生成热力图')
    parser.add_argument('--max_per_class', type=int, default=10, 
                        help='每个类别最多生成的热力图数量')
    parser.add_argument('--max_total', type=int, default=50, 
                        help='总共最多生成的热力图数量')
    
    # 优化版ResNet50专用参数
    parser.add_argument('--mining_ratio', type=float, default=0.25,
                       help='困难样本挖掘比例 (仅resnet50_optimized)')
    parser.add_argument('--triangular_margin', type=float, default=0.8,
                       help='三角损失边界 (仅resnet50_optimized)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='梯度裁剪阈值')
    
    # 其他参数
    parser.add_argument('--eval_only', action='store_true', help='仅进行评估，不训练')
    parser.add_argument('--checkpoint', type=str, default=None, help='用于评估的检查点路径')
    
    return parser.parse_args()

class MultiStageScheduler:
    """性能自适应的多阶段学习率调度器
    根据模型性能动态调整学习率策略，而非固定轮次
    """
    def __init__(self, optimizer, num_epochs, init_lr, min_lr):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # 各阶段最小轮次数要求 - 增加最小轮次要求
        self.min_epochs_per_stage = {
            'warmup': max(5, int(num_epochs * 0.05)),  # 至少5轮或总轮次的5%
            'aggressive': max(15, int(num_epochs * 0.15)),  # 至少15轮或总轮次的15%
            'refinement': max(20, int(num_epochs * 0.2)),  # 至少20轮或总轮次的20%
            'fine_tuning': max(10, int(num_epochs * 0.1))   # 至少10轮或总轮次的10%
        }
        
        # 当前所处阶段
        self.current_stage = 'warmup'
        self.current_stage_epochs = 0
        
        # 用于跟踪性能和学习率变化
        self.accuracy_history = []
        self.lr_history = []
        self.best_accuracy = 0
        self.plateau_counter = 0
        
        # 阶段切换条件 - 增加容忍度
        self.stagnation_threshold = 5  # 性能停滞轮次阈值增加到5
        self.improvement_threshold = 0.002  # 性能提升阈值降低，更容易检测到进步
        
        # 各阶段使用的调度器
        self.schedulers = {}
        self._setup_schedulers()
        
        # 记录上一个阶段结束时的学习率
        self.last_stage_lr = init_lr
        
        # 设置学习率保护机制
        self.lr_protection = {
            'max_decrease_factor': 0.5,  # 单次最大下降比例
            'min_lr_factor': 0.1,  # 相对于初始学习率的最小值比例
            'recovery_factor': 1.2  # 性能提升时的恢复系数
        }
        
        # 记录各个阶段的边界，用于可视化
        self.stage_boundaries = {
            'warmup_end': 0,
            'aggressive_end': 0,
            'refinement_end': 0
        }
        
        # 平滑过渡机制
        self.transition_steps = 3  # 阶段过渡的平滑步数
        self.in_transition = False
        self.transition_from = None
        self.transition_to = None
        self.transition_step = 0
        
    def _setup_schedulers(self):
        """设置各阶段的学习率调度器"""
        # 预热阶段：线性增加到初始学习率，更平缓
        self.schedulers['warmup'] = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.2,  # 提高起始因子
            end_factor=1.0,
            total_iters=self.min_epochs_per_stage['warmup']
        )
        
        # 积极探索阶段：1cycle政策，参数更加平滑
        self.schedulers['aggressive'] = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.init_lr * 1.5,  # 降低最大学习率，防止过大波动
            total_steps=self.min_epochs_per_stage['aggressive'] * 2,
            pct_start=0.4,  # 增加上升比例
            anneal_strategy='cos',
            div_factor=5.0,  # 减小初始下降幅度
            final_div_factor=4.0  # 减小最终下降幅度
        )
        
        # 细化阶段：余弦退火，使用更大的周期和更高的最小值
        self.schedulers['refinement'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.min_epochs_per_stage['refinement'] * 3,  # 增加周期长度
            eta_min=self.min_lr * 20  # 增加最小学习率
        )
        
    def step(self, accuracy=None):
        """更新学习率，基于当前性能动态调整阶段"""
        # 更新当前轮次
        self.current_epoch += 1
        self.current_stage_epochs += 1
        
        # 记录准确率历史
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
            
            # 更新最佳准确率和停滞计数器
            if accuracy > self.best_accuracy + 0.0005:  # 添加小容差
                self.best_accuracy = accuracy
                self.plateau_counter = 0
            else:
                self.plateau_counter += 1
                
        # 如果在阶段过渡中，处理平滑过渡
        if self.in_transition:
            result = self._handle_transition()
            if result:  # 如果过渡完成
                self.in_transition = False
                self.current_stage = self.transition_to
                self.transition_to = None
                self.transition_from = None
                self.transition_step = 0
                self.current_stage_epochs = 0
                self.plateau_counter = 0
        # 检查是否应该开始新的阶段过渡
        elif self._should_switch_stage():
            next_stage = self._get_next_stage()
            if next_stage != self.current_stage:
                # 记录当前阶段结束轮次，用于可视化
                if self.current_stage == 'warmup':
                    self.stage_boundaries['warmup_end'] = self.current_epoch
                elif self.current_stage == 'aggressive':
                    self.stage_boundaries['aggressive_end'] = self.current_epoch
                elif self.current_stage == 'refinement':
                    self.stage_boundaries['refinement_end'] = self.current_epoch
                
                # 启动阶段过渡
                self.in_transition = True
                self.transition_from = self.current_stage
                self.transition_to = next_stage
                self.transition_step = 0
        
        # 根据当前阶段更新学习率
        if not self.in_transition:
            if self.current_stage in ['warmup', 'aggressive', 'refinement']:
                # 使用预定义的调度器
                self.schedulers[self.current_stage].step()
            else:
                # 微调阶段：基于验证准确率自适应调整学习率
                self._adaptive_step(accuracy)
        
        # 应用学习率保护机制
        self._apply_lr_protection()
        
        # 记录当前学习率
        current_lr = self.get_last_lr()
        self.lr_history.append(current_lr[0])
        
        return current_lr
    
    def _handle_transition(self):
        """处理阶段之间的平滑过渡"""
        self.transition_step += 1
        
        # 获取源阶段和目标阶段的学习率
        if self.transition_from in ['warmup', 'aggressive', 'refinement']:
            from_lr = self.schedulers[self.transition_from].get_last_lr()[0]
        else:
            from_lr = self.get_last_lr()[0]
            
        # 计算目标学习率
        if self.transition_to == 'aggressive':
            # 积极阶段的起始学习率应该与上一阶段结束时接近
            to_lr = from_lr * 1.1  # 轻微增加以开始探索
        elif self.transition_to == 'refinement':
            # 细化阶段应该从当前学习率开始
            to_lr = from_lr
        elif self.transition_to == 'fine_tuning':
            # 微调阶段应该使用较低但不至于太低的学习率
            to_lr = max(from_lr * 0.7, self.min_lr * 30)
        else:
            to_lr = from_lr
            
        # 使用余弦过渡计算当前学习率
        progress = self.transition_step / self.transition_steps
        transition_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        current_lr = to_lr + (from_lr - to_lr) * transition_factor
        
        # 应用过渡学习率
        for group in self.optimizer.param_groups:
            group['lr'] = current_lr
            
        # 如果过渡完成，初始化下一阶段
        if self.transition_step >= self.transition_steps:
            self._initialize_next_stage(self.transition_to, current_lr)
            return True
            
        return False
            
    def _initialize_next_stage(self, new_stage, current_lr):
        """初始化新阶段的调度器"""
        self.last_stage_lr = current_lr
        
        # 根据新阶段重新初始化调度器
        if new_stage == 'aggressive':
            self.schedulers['aggressive'] = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=current_lr * 1.5,  # 相对当前学习率的最大值
                total_steps=self.min_epochs_per_stage['aggressive'] * 2,
                pct_start=0.4,
                anneal_strategy='cos',
                div_factor=3.0,  # 减小波动
                final_div_factor=3.0
            )
        elif new_stage == 'refinement':
            self.schedulers['refinement'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.min_epochs_per_stage['refinement'] * 3,
                eta_min=max(self.min_lr * 20, current_lr * 0.2)  # 确保最小值不会太低
            )
    
    def _should_switch_stage(self):
        """检查是否应该切换到下一个阶段"""
        # 没有足够的准确率历史数据
        if len(self.accuracy_history) < 5:  # 增加到至少5个样本
            return False
            
        # 检查是否满足最小轮次要求
        if self.current_stage_epochs < self.min_epochs_per_stage[self.current_stage]:
            return False
            
        # 根据当前阶段设置不同的切换策略
        if self.current_stage == 'warmup':
            # 预热阶段：最近5轮的平均准确率提升小于阈值时切换
            if len(self.accuracy_history) < 7:
                return False
                
            recent_improvements = [self.accuracy_history[i] - self.accuracy_history[i-1] 
                                 for i in range(len(self.accuracy_history)-5, len(self.accuracy_history))]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            return avg_improvement < 0.01 and avg_improvement > 0  # 增长放缓但仍为正
            
        elif self.current_stage == 'aggressive':
            # 积极探索阶段：准确率趋于稳定或达到较高水平时切换
            recent_improvements = [self.accuracy_history[i] - self.accuracy_history[i-1] 
                                 for i in range(len(self.accuracy_history)-4, len(self.accuracy_history))]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            
            # 当准确率增长放缓或者已经达到较高水平时切换
            high_accuracy_threshold = 0.85  # 高准确率阈值
            return (avg_improvement < self.improvement_threshold or 
                   (self.accuracy_history[-1] > high_accuracy_threshold and self.current_stage_epochs > self.min_epochs_per_stage['aggressive'] * 1.5))
            
        elif self.current_stage == 'refinement':
            # 细化阶段：出现较长时间停滞或接近最大轮次时切换
            max_epochs_factor = 1.5  # 最大轮次因子
            approaching_max_epochs = self.current_stage_epochs > self.min_epochs_per_stage['refinement'] * max_epochs_factor
            return self.plateau_counter >= self.stagnation_threshold or approaching_max_epochs
            
        return False
    
    def _get_next_stage(self):
        """获取下一个阶段"""
        if self.current_stage == 'warmup':
            return 'aggressive'
        elif self.current_stage == 'aggressive':
            return 'refinement'
        elif self.current_stage == 'refinement':
            return 'fine_tuning'
        return 'fine_tuning'
    
    def _transition_to_stage(self, new_stage):
        """处理阶段切换时的学习率过渡"""
        # 此方法已被_handle_transition和_initialize_next_stage替代
        pass
    
    def _adaptive_step(self, accuracy):
        """微调阶段的自适应学习率调整"""
        if accuracy is None:
            return
        
        # 跟踪准确率历史以检测趋势
        window_size = min(5, len(self.accuracy_history))
        if window_size < 3:
            return
            
        recent_accuracies = self.accuracy_history[-window_size:]
        
        # 计算趋势 - 线性回归斜率
        x = list(range(window_size))
        slope = sum((x[i] - sum(x)/window_size) * (recent_accuracies[i] - sum(recent_accuracies)/window_size) 
                  for i in range(window_size)) / sum((x[i] - sum(x)/window_size)**2 for i in range(window_size))
        
        current_lr = self.get_last_lr()[0]
        
        # 根据趋势调整学习率
        if slope < -0.001:  # 明显下降趋势
            # 提高学习率以跳出局部最小值
            new_lr = min(current_lr * 2.0, self.init_lr * 0.1)
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr
            self.plateau_counter = 0
            
        elif abs(slope) < 0.0005:  # 平稳期
            if self.plateau_counter >= 3:  # 持续平稳
                # 温和降低学习率
                new_lr = max(current_lr * 0.8, self.min_lr)
                for group in self.optimizer.param_groups:
                    group['lr'] = new_lr
                self.plateau_counter = 0
            
        else:  # 上升趋势，保持当前学习率
            pass
    
    def _apply_lr_protection(self):
        """应用学习率保护机制，防止学习率突然下降或过低"""
        if not self.lr_history:
            return
            
        previous_lr = self.lr_history[-1] if len(self.lr_history) > 0 else self.init_lr
        current_lr = self.get_last_lr()[0]
        
        # 防止单次下降过多
        if current_lr < previous_lr * self.lr_protection['max_decrease_factor']:
            protected_lr = previous_lr * self.lr_protection['max_decrease_factor']
            for group in self.optimizer.param_groups:
                group['lr'] = protected_lr
                
        # 确保学习率不会低于最小保护值
        min_protected_lr = self.init_lr * self.lr_protection['min_lr_factor']
        if current_lr < min_protected_lr and self.current_stage != 'fine_tuning':
            for group in self.optimizer.param_groups:
                group['lr'] = min_protected_lr
                
        # 如果性能提升，给予学习率恢复机会
        if len(self.accuracy_history) >= 2 and self.accuracy_history[-1] > self.accuracy_history[-2] + 0.005:
            current_lr = self.get_last_lr()[0]  # 获取可能已被保护的当前学习率
            if current_lr < previous_lr and self.current_stage == 'fine_tuning':
                recovery_lr = min(current_lr * self.lr_protection['recovery_factor'], previous_lr)
                for group in self.optimizer.param_groups:
                    group['lr'] = recovery_lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_stage_info(self):
        """获取当前阶段信息，用于日志记录"""
        stage = self.transition_to if self.in_transition else self.current_stage
        stage_display = f"{self.transition_from}->{stage} (过渡中 {self.transition_step}/{self.transition_steps})" if self.in_transition else stage
        
        return {
            'stage': stage_display,
            'epoch_in_stage': self.current_stage_epochs,
            'min_epochs': self.min_epochs_per_stage[self.current_stage],
            'learning_rate': self.get_last_lr()[0],
            'stage_boundaries': self.stage_boundaries,
            'plateau_counter': self.plateau_counter,
            'in_transition': self.in_transition
        }

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置matplotlib，检查是否支持中文
    use_chinese = setup_matplotlib()
    
    # 初始化配置参数
    config = {
        'data_dir': args.data_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'min_delta': 0.001,
        'warmup_epochs': 5,
        'accumulation_steps': args.accumulation_steps,
        'temperature': 2.0,
        'contrast_temperature': 0.1,
        'memory_bank_size': 4096,
        'gradient_clip': 1.0,
        'min_lr': 1e-6,
        'accuracy_threshold': args.accuracy_threshold,
        'max_per_class': args.max_per_class,
        'max_total': args.max_total,
        'use_chinese': use_chinese,  # 添加中文支持标志
        'eval_frequency': 20,  # 每20个epoch评估一次
        'model_type': args.model_type,  # 模型类型
        'use_attention': not args.no_attention,  # 是否使用注意力机制
        'scheduler_type': args.scheduler_type,  # 学习率调度器类型
        'cosine_T_max': args.cosine_T_max if args.cosine_T_max else args.num_epochs,  # 余弦调度器的T_max参数
        'cosine_eta_min': args.cosine_eta_min,  # 余弦调度器的最小学习率
        'cosine_warmup_epochs': args.cosine_warmup_epochs,  # 余弦调度器的预热轮次
        # 优化版ResNet50专用参数
        'mining_ratio': args.mining_ratio,  # 困难样本挖掘比例
        'triangular_margin': args.triangular_margin,  # 三角损失边界
        'gradient_clip': args.gradient_clip  # 梯度裁剪阈值
    }
    
    # 如果使用优化版ResNet50，自动调整部分参数
    if config['model_type'] == 'resnet50_optimized':
        print("检测到优化版ResNet50，自动应用优化配置...")
        
        # 如果batch_size较大，自动减小以适应更复杂的模型
        if config['batch_size'] > 16:
            original_batch_size = config['batch_size']
            config['batch_size'] = max(8, config['batch_size'] // 2)
            print(f"自动调整batch_size: {original_batch_size} -> {config['batch_size']}")
        
        # 如果学习率较高，自动降低
        if config['learning_rate'] > 5e-5:
            original_lr = config['learning_rate']
            config['learning_rate'] = config['learning_rate'] * 0.8
            print(f"自动调整学习率: {original_lr} -> {config['learning_rate']}")
        
        # 增加权重衰减
        original_weight_decay = config['weight_decay']
        config['weight_decay'] = config['weight_decay'] * 1.2
        print(f"自动调整权重衰减: {original_weight_decay} -> {config['weight_decay']}")
        
        # 增加耐心值
        config['patience'] = config['patience'] + 5
        print(f"增加early stopping耐心值到: {config['patience']}")
        
        # 确保梯度累积步数至少为2
        if config['accumulation_steps'] < 2:
            config['accumulation_steps'] = 2
            print(f"设置梯度累积步数为: {config['accumulation_steps']}")
    
    # 如果提供了配置文件，从中加载参数
    if args.config and os.path.exists(args.config):
        try:
            print(f"正在从配置文件加载参数: {args.config}")
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                # 更新配置参数
                config.update(file_config)
                print(f"已成功加载配置文件")
        except Exception as e:
            print(f"加载配置文件时出错: {str(e)}")
    
    # 创建保存目录
    model_name = config['model_type']
    if not config['use_attention'] and model_name != 'ensemble':
        model_name += '_no_attention'
    model_name += f"_{config['scheduler_type']}"
    
    save_dir = config.get('checkpoint_dir', f'checkpoints/{model_name}')
    log_dir = config.get('log_dir', f'logs/{model_name}')
    visualization_dir = config.get('visualization_dir', f'visualization/{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # 如果指定了特定GPU，使用指定的GPU
    if 'cuda_device' in config and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['cuda_device']}")
    
    # 优化CUDA设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置日志
    logger = setup_logger(log_dir)
    logger.info(f'Using device: {device}')
    logger.info('配置参数:')
    for k, v in config.items():
        logger.info(f'{k}: {v}')
    
    try:
        # 加载数据和类别映射
        with open(os.path.join(config['data_dir'], 'class_mapping.json'), 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
            num_classes = len(class_mapping)
            # 创建反向映射（id到名称）
            id_to_name = {v: k for k, v in class_mapping.items()}
        
        train_loader, val_loader = create_data_loaders(
            config['data_dir'],
            config['batch_size'],
            num_workers=config.get('num_workers', min(8, os.cpu_count()))
        )
        
        # 创建模型并应用配置参数
        if config['model_type'] == 'ensemble':
            model = EnsembleModel(
                num_classes=num_classes,
                temperature=config.get('temperature', 2.0)
            ).to(device)
        else:
            # 使用我们新创建的模型函数
            if config['use_attention']:
                model = create_model(
                    config['model_type'],
                    num_classes=num_classes
                ).to(device)
            else:
                model = create_model_without_attention(
                    config['model_type'],
                    num_classes=num_classes
                ).to(device)
        
        # 输出选择的模型信息
        logger.info(f"使用模型: {config['model_type']}, 注意力机制: {config['use_attention']}")
        print(f"使用模型: {config['model_type']}, 注意力机制: {config['use_attention']}")
        
        # 如果配置了模型参数，应用这些参数
        if 'dropout_rate' in config:
            # 更新Dropout率
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = config['dropout_rate']
        
        if 'inception_freeze_layers' in config:
            # 更新Inception模型冻结层数
            freeze_layers = int(config['inception_freeze_layers'])
            for param in list(model.inception.inception.parameters())[:-freeze_layers]:
                param.requires_grad = False
        
        if 'fpn_channels' in config:
            # 更新FPN通道数
            fpn_channels = int(config['fpn_channels'])
            model.inception.fpn.out_channels = fpn_channels
            model.efficientnet.fpn.out_channels = fpn_channels
        
        # 更新损失函数参数
        if hasattr(model, 'combined_loss') and hasattr(model.combined_loss, 'focal_loss'):
            if 'focal_gamma' in config:
                model.combined_loss.focal_loss.gamma = config['focal_gamma']
            if 'label_smoothing' in config:
                model.combined_loss.focal_loss.smoothing = config['label_smoothing']
        
        if hasattr(model, 'kd_loss') and 'temperature' in config:
            model.kd_loss.temperature = config['temperature']
        
        # 如果只进行评估
        if args.eval_only:
            if args.checkpoint:
                checkpoint_path = args.checkpoint
            else:
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                
            if os.path.exists(checkpoint_path):
                print(f"\n加载检查点: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                # 使用strict=False允许加载部分权重，忽略不匹配的键
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("模型加载成功，部分层可能未加载（模型结构可能已更新）")
                
                # 进行评估
                final_results = validate(model, val_loader, device)
                
                # 输出评估结果
                print("\n评估结果:")
                print(f"准确率: {final_results['accuracy']*100:.2f}%")
                print(f"宏平均F1: {final_results['macro_f1']:.4f}")
                print(f"加权平均F1: {final_results['weighted_f1']:.4f}")
                
                # 生成混淆矩阵
                visualize_confusion_matrix(
                    final_results['confusion_matrix'],
                    list(class_mapping.keys()),
                    epoch=999,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                
                # 生成性能报告
                final_report_file = analyze_performance(
                    final_results,
                    class_mapping,
                    epoch=999,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                print(f"性能报告已保存到: {final_report_file}")
                
                # 生成热力图
                print("\n开始生成低准确率类别的热力图...")
                generate_low_accuracy_class_heatmaps(
                    model, 
                    val_loader, 
                    device, 
                    class_mapping,
                    accuracy_threshold=config['accuracy_threshold'],
                    max_per_class=config['max_per_class'],
                    max_total=config['max_total'],
                    save_dir=visualization_dir
                )
                
                # 分析混淆矩阵，找出最容易混淆的类别对
                print("\n开始分析混淆矩阵，找出最容易混淆的类别对...")
                confusion_pairs_file = analyze_confusion_pairs(
                    final_results['confusion_matrix'],
                    class_mapping,
                    top_n=20,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                print(f"混淆矩阵分析已保存到: {confusion_pairs_file}")
                
                return
            else:
                print(f"错误: 检查点文件不存在: {checkpoint_path}")
                return
        
        # 优化器 - 使用配置参数
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('eps', 1e-8)
        )
        
        # 学习率调度器选择
        if config['scheduler_type'] == 'multistage':
            # 使用多阶段学习率调度器
            scheduler = MultiStageScheduler(
                optimizer,
                config['num_epochs'],
                config['learning_rate'],
                config.get('min_lr', 1e-6)
            )
            logger.info(f"使用多阶段学习率调度器")
            print(f"使用多阶段学习率调度器")
        else:
            # 使用单一余弦学习率调度器，带预热
            if config['cosine_warmup_epochs'] > 0:
                # 线性预热 + 余弦退火
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, 
                    start_factor=0.1, 
                    end_factor=1.0,
                    total_iters=config['cosine_warmup_epochs']
                )
                
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config['num_epochs'] - config['cosine_warmup_epochs'],
                    eta_min=config['cosine_eta_min']
                )
                
                # 顺序调度器
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[config['cosine_warmup_epochs']]
                )
            else:
                # 只使用余弦退火
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config['cosine_T_max'],
                    eta_min=config['cosine_eta_min']
                )
            
            logger.info(f"使用单一余弦学习率调度器，T_max={config['cosine_T_max']}，预热轮次={config['cosine_warmup_epochs']}")
            print(f"使用单一余弦学习率调度器，T_max={config['cosine_T_max']}，预热轮次={config['cosine_warmup_epochs']}")
        
        # 使用混合精度训练
        scaler = GradScaler(
            init_scale=config.get('init_scale', 2**10),
            growth_factor=config.get('growth_factor', 2.0),
            backoff_factor=config.get('backoff_factor', 0.5),
            growth_interval=config.get('growth_interval', 100)
        )
        
        # 早停机制
        early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=config.get('min_delta', 0.001)
        )
        
        # 损失监控
        loss_monitor = LossMonitor()
        
        # 训练循环
        best_val_acc = 0.0
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_loss': [],
            'learning_rates': [],
            'class_acc': {},
            'macro_f1': [],
            'weighted_f1': [],
            'class_f1': {},
            'class_precision': {},
            'class_recall': {}
        }
        
        # 存储当前类别准确率的字典
        current_class_accuracies = None
        
        for epoch in range(config['num_epochs']):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}]")
            print(f"\n{'='*50}")
            print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
            
            # 设置数据增强参数（确保相关变换存在）
            try:
                # 应用MixUp和RandomErasing参数
                if hasattr(train_loader.dataset, 'transform'):
                    for transform in train_loader.dataset.transform.transforms:
                        # 更新MixUp alpha
                        if hasattr(transform, 'alpha') and 'mixup_alpha' in config:
                            transform.alpha = config['mixup_alpha']
                        # 更新RandomErasing概率
                        if hasattr(transform, 'p') and hasattr(transform, 'value') and 'random_erasing_prob' in config:
                            transform.p = config['random_erasing_prob']
            except Exception as e:
                logger.warning(f"设置数据增强参数失败: {str(e)}")
            
            # 训练，传入当前类别准确率
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer,
                scaler, device, epoch, config['num_epochs'],
                config.get('gradient_clip', 1.0),
                class_accuracies=current_class_accuracies,
                accumulation_steps=config.get('accumulation_steps', 1)
            )
            
            # 验证
            val_results = validate(model, val_loader, device)
            
            # 更新当前类别准确率
            current_class_accuracies = val_results['class_accuracies']
            
            # 更新学习率
            if config['scheduler_type'] == 'multistage':
                current_lr = scheduler.step(val_results['accuracy'])
                stage_info = scheduler.get_stage_info()
            else:
                scheduler.step()
                current_lr = [group['lr'] for group in optimizer.param_groups]
                stage_info = {'stage': 'cosine', 'epoch_in_stage': epoch, 'plateau_counter': 0}
            
            # 更新训练历史
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_acc'].append(val_results['accuracy'])
            train_history['val_loss'].append(val_results['loss'])
            train_history['learning_rates'].append(current_lr[0])
            train_history['macro_f1'].append(val_results['macro_f1'])
            train_history['weighted_f1'].append(val_results['weighted_f1'])
            
            # 添加阶段信息到训练历史
            if 'lr_stages' not in train_history:
                train_history['lr_stages'] = []
            
            if config['scheduler_type'] == 'multistage':
                train_history['lr_stages'].append(stage_info['stage'])
                
                if 'stage_boundaries' not in train_history:
                    train_history['stage_boundaries'] = stage_info['stage_boundaries']
                else:
                    train_history['stage_boundaries'] = stage_info['stage_boundaries']
            else:
                train_history['lr_stages'].append('cosine')
            
            # 更新每个类别的评估指标历史
            for class_id, accuracy in val_results['class_accuracies'].items():
                class_name = id_to_name[class_id]
                
                # 更新准确率历史
                if class_name not in train_history['class_acc']:
                    train_history['class_acc'][class_name] = []
                train_history['class_acc'][class_name].append(accuracy)
                
                # 更新F1分数历史
                if class_name not in train_history['class_f1']:
                    train_history['class_f1'][class_name] = []
                train_history['class_f1'][class_name].append(val_results['class_f1'][class_id])
                
                # 更新精确率历史
                if class_name not in train_history['class_precision']:
                    train_history['class_precision'][class_name] = []
                train_history['class_precision'][class_name].append(val_results['class_precision'][class_id])
                
                # 更新召回率历史
                if class_name not in train_history['class_recall']:
                    train_history['class_recall'][class_name] = []
                train_history['class_recall'][class_name].append(val_results['class_recall'][class_id])
            
            # 输出训练信息
            print(f"训练准确率: {train_acc*100:.2f}%")
            print(f"验证准确率: {val_results['accuracy']*100:.2f}%")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_results['loss']:.4f}")
            print(f"宏平均F1: {val_results['macro_f1']:.4f}")
            print(f"加权平均F1: {val_results['weighted_f1']:.4f}")
            print(f"当前学习率: {current_lr[0]:.6f} (阶段: {stage_info['stage']}, 当前阶段第{stage_info['epoch_in_stage']}轮, 停滞计数: {stage_info['plateau_counter']})")
            print(f"{'='*50}\n")
            
            # 记录到日志
            logger.info(f"\n训练统计:")
            logger.info(f"学习率: {current_lr[0]:.6f} (阶段: {stage_info['stage']}, 当前阶段第{stage_info['epoch_in_stage']}轮)")
            logger.info(f"训练损失: {train_loss:.4f}")
            logger.info(f"验证损失: {val_results['loss']:.4f}")
            logger.info(f"训练准确率: {train_acc*100:.2f}%")
            logger.info(f"验证准确率: {val_results['accuracy']*100:.2f}%")
            logger.info(f"宏平均F1: {val_results['macro_f1']:.4f}")
            logger.info(f"加权平均F1: {val_results['weighted_f1']:.4f}")
            
            # 只在指定频率生成混淆矩阵和性能分析
            if epoch % config['eval_frequency'] == 0 or epoch == config['num_epochs'] - 1:
                # 可视化混淆矩阵
                visualize_confusion_matrix(
                    val_results['confusion_matrix'], 
                    list(class_mapping.keys()), 
                    epoch,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                
                # 分析性能
                performance_report_file = analyze_performance(
                    val_results, 
                    class_mapping, 
                    epoch,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                logger.info(f'性能报告已保存到: {performance_report_file}')
            
            # 保存最佳模型
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.get_stage_info() if hasattr(scheduler, 'get_stage_info') else None,
                    'val_acc': val_results['accuracy'],
                    'val_loss': val_results['loss'],
                    'macro_f1': val_results['macro_f1'],
                    'weighted_f1': val_results['weighted_f1'],
                    'class_acc': val_results['class_accuracies'],
                    'confusion_matrix': val_results['confusion_matrix'],
                    'train_history': train_history
                }, os.path.join(save_dir, 'best_model.pth'))
                logger.info("最佳模型已保存")
            
            # 检查早停
            early_stopping(train_loss, val_results['accuracy'], epoch)
            if early_stopping.early_stop:
                print(f"\n[早停] 触发早停机制，停止训练")
                print(f"最佳验证准确率: {best_val_acc*100:.2f}% (第 {early_stopping.best_epoch + 1} 轮)")
                print(f"当前轮次: {epoch + 1}")
                print(f"已经 {config['patience']} 轮未提升，停止训练")
                logger.info('\n触发早停机制，停止训练')
                logger.info(f"最佳验证准确率: {best_val_acc*100:.2f}% (第 {early_stopping.best_epoch + 1} 轮)")
                break
        
        print(f"\n训练完成! 最佳验证准确率: {best_val_acc*100:.2f}%")
        
        # 保存训练历史
        history_file = os.path.join(log_dir, 'ensemble_training_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(train_history, f, ensure_ascii=False, indent=4)
        logger.info(f'训练历史已保存到: {history_file}')
        
        # 绘制学习曲线
        plot_learning_curves(
            train_history, 
            save_dir=visualization_dir,
            use_chinese=config['use_chinese']
        )
        logger.info('学习曲线已生成')
        
        # 加载最佳模型进行最终评估
        print("\n加载最佳模型进行最终评估...")
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, weights_only=False)
            # 使用strict=False允许加载部分权重，忽略不匹配的键
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("模型加载成功，部分层可能未加载（模型结构可能已更新）")
            
            # 进行最终评估
            final_results = validate(model, val_loader, device)
            
            # 生成最终混淆矩阵
            visualize_confusion_matrix(
                final_results['confusion_matrix'],
                list(class_mapping.keys()),
                epoch=999,  # 使用特殊值表示最终结果
                save_dir=visualization_dir,
                use_chinese=config['use_chinese']
            )
            
            # 生成最终性能报告
            final_report_file = analyze_performance(
                final_results,
                class_mapping,
                epoch=999,  # 使用特殊值表示最终结果
                save_dir=visualization_dir,
                use_chinese=config['use_chinese']
            )
            
            # 输出最终结果
            print("\n最终评估结果:")
            print(f"准确率: {final_results['accuracy']*100:.2f}%")
            print(f"宏平均F1: {final_results['macro_f1']:.4f}")
            print(f"加权平均F1: {final_results['weighted_f1']:.4f}")
            print(f"最终性能报告已保存到: {final_report_file}")
            
            # 记录到日志
            logger.info("\n最终评估结果:")
            logger.info(f"准确率: {final_results['accuracy']*100:.2f}%")
            logger.info(f"宏平均F1: {final_results['macro_f1']:.4f}")
            logger.info(f"加权平均F1: {final_results['weighted_f1']:.4f}")
            
            # 生成低准确率类别的错误预测图像的热力图
            print("\n开始生成低准确率类别的热力图...")
            generate_low_accuracy_class_heatmaps(
                model, 
                val_loader, 
                device, 
                class_mapping,
                accuracy_threshold=config['accuracy_threshold'],
                max_per_class=config['max_per_class'],
                max_total=config['max_total'],
                save_dir=visualization_dir
            )
            print("热力图生成完成！")
            
            # 分析混淆矩阵，找出最容易混淆的类别对
            print("\n开始分析混淆矩阵，找出最容易混淆的类别对...")
            confusion_pairs_file = analyze_confusion_pairs(
                final_results['confusion_matrix'],
                class_mapping,
                top_n=20,
                save_dir=visualization_dir,
                use_chinese=config['use_chinese']
            )
            logger.info(f'混淆矩阵分析已保存到: {confusion_pairs_file}')
        else:
            logger.warning(f"未找到最佳模型文件: {best_model_path}")
            
    except Exception as e:
        logger.error(f'训练过程中发生错误: {str(e)}')
        raise e

if __name__ == "__main__":
    main() 