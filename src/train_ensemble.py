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

# Resolve OpenMP library conflict issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def setup_matplotlib():
    """Configure matplotlib to support Chinese font display"""
    import matplotlib
    # Force use of Agg backend to avoid GUI-related issues
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # Try to set up Chinese fonts
    try:
        # Try to use Chinese fonts available in the system
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
        
        # Check if Chinese fonts were successfully set
        fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = [f for f in fonts if '黑体' in f or '雅黑' in f or 'SimSun' in f or 'SimHei' in f]
        
        if not chinese_fonts:
            # If no Chinese fonts found, use English
            print("Warning: No Chinese fonts found, will use English display")
            use_chinese = False
        else:
            print(f"Found Chinese font: {chinese_fonts[0]}")
            use_chinese = True
    except Exception as e:
        print(f"Error setting up Chinese fonts: {str(e)}, will use English display")
        use_chinese = False
    
    return use_chinese

def setup_logger(log_dir):
    """Set up logging"""
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
    """Early stopping mechanism"""
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
        Check if early stopping should be triggered
        
        Args:
            val_loss: Validation loss
            val_acc: Validation accuracy
            epoch: Current epoch
        
        Returns:
            bool: Whether early stopping should be triggered
        """
        # Initialize best_acc on first call
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_epoch = epoch
            return False
        # If validation accuracy improves, update best accuracy and epoch
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
        # If validation accuracy hasn't improved for patience consecutive epochs, trigger early stopping
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
                print("Detected consecutive invalid losses, restoring to last valid state")
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
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Calculate current domain adaptation weight
    alpha = 2.0 / (1.0 + np.exp(-10 * epoch / total_epochs)) - 1.0
    
    # Calculate current contrastive learning weight
    contrast_weight = min(0.1 * (epoch / 10), 0.5)  # Gradually increase contrastive learning weight
    
    with tqdm(train_loader, desc='Training', ncols=100) as pbar:
        for idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(device_type='cuda:1' if torch.cuda.is_available() else 'cpu'):
                # Determine whether to pass contrast_weight parameter based on model type
                if isinstance(model, EnsembleModel):
                    # EnsembleModel supports contrast_weight parameter
                    outputs, loss = model(
                        inputs, 
                        labels, 
                        alpha=alpha,
                        class_accuracies=class_accuracies,
                        contrast_weight=contrast_weight
                    )
                else:
                    # Other models do not support contrast_weight parameter
                    outputs, loss = model(
                        inputs, 
                        labels, 
                        alpha=alpha,
                        class_accuracies=class_accuracies
                    )
                
                # Check if loss value is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss value detected: {loss}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            
            # Backward propagation
            scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0:
                # Add gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/(idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'alpha': f'{alpha:.3f}'
            })
            
            if idx % 500 == 0:
                torch.cuda.empty_cache()
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, device):
    """Validate model performance"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    print("\nStarting validation...")
    with torch.no_grad():
        # Use tqdm to create progress bar, set leave=True to keep progress bar
        for inputs, labels in tqdm(val_loader, desc="Validation progress", leave=True, ncols=100):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Determine autocast device_type based on device type
            device_type = 'cuda:1' if device.type == 'cuda:1' else 'cpu'
            
            with autocast(device_type=device_type):
                outputs = model(inputs)
                # Handle case where outputs is a tuple, take first element as logits
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    # Calculate confusion matrix
    num_classes = len(np.unique(all_labels))
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        confusion_mat[t, p] += 1
    
    # Calculate accuracy for each class
    class_accuracies = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    class_support = {}
    
    for class_id in range(num_classes):
        # Total number of samples for this class
        class_total = np.sum(all_labels == class_id)
        if class_total > 0:
            # Number of correctly classified samples for this class
            class_correct = confusion_mat[class_id, class_id]
            # Accuracy = correctly classified samples / total samples for this class
            class_accuracies[class_id] = class_correct / class_total
            
            # Calculate precision = TP / (TP + FP)
            # TP = number of samples correctly predicted as this class
            # FP = number of samples incorrectly predicted as this class
            predicted_as_class = np.sum(confusion_mat[:, class_id])
            if predicted_as_class > 0:
                class_precision[class_id] = confusion_mat[class_id, class_id] / predicted_as_class
            else:
                class_precision[class_id] = 0.0
            
            # Calculate recall = TP / (TP + FN)
            # TP = number of samples correctly predicted as this class
            # FN = number of samples of this class incorrectly predicted as other classes
            class_recall[class_id] = confusion_mat[class_id, class_id] / class_total
            
            # Calculate F1 score = 2 * (precision * recall) / (precision + recall)
            if class_precision[class_id] + class_recall[class_id] > 0:
                class_f1[class_id] = 2 * (class_precision[class_id] * class_recall[class_id]) / (class_precision[class_id] + class_recall[class_id])
            else:
                class_f1[class_id] = 0.0
                
            # Record support (number of samples)
            class_support[class_id] = int(class_total)
    
    # Calculate macro average and weighted average metrics
    macro_precision = np.mean(list(class_precision.values()))
    macro_recall = np.mean(list(class_recall.values()))
    macro_f1 = np.mean(list(class_f1.values()))
    
    # Calculate weighted average, considering sample count for each class
    weights = np.array([class_support[i] for i in range(num_classes)])
    weights = weights / np.sum(weights)
    
    weighted_precision = np.sum([class_precision[i] * weights[i] for i in range(num_classes)])
    weighted_recall = np.sum([class_recall[i] * weights[i] for i in range(num_classes)])
    weighted_f1 = np.sum([class_f1[i] * weights[i] for i in range(num_classes)])
    
    # Return validation results
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
    """Generate heatmaps for low accuracy classes"""
    print("\nStarting to generate heatmaps for low accuracy classes...")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create reverse mapping (id to name)
    id_to_name = {v: k for k, v in class_mapping.items()}
    
    # First perform validation to get accuracy for each class
    val_results = validate(model, val_loader, device)
    class_accuracies = val_results['class_accuracies']
    
    # Find classes with accuracy below threshold
    low_accuracy_classes = {class_id: acc for class_id, acc in class_accuracies.items() if acc < accuracy_threshold}
    
    # If no classes below threshold, select the 10 classes with lowest accuracy
    if not low_accuracy_classes:
        print(f"No classes found with accuracy below {accuracy_threshold*100:.1f}%, will select the 10 classes with lowest accuracy")
        sorted_accuracies = sorted(class_accuracies.items(), key=lambda x: x[1])
        low_accuracy_classes = {class_id: acc for class_id, acc in sorted_accuracies[:10]}
    
    print(f"Found {len(low_accuracy_classes)} low accuracy classes:")
    for class_id, accuracy in low_accuracy_classes.items():
        class_name = id_to_name[class_id]
        print(f"  - {class_name}: {accuracy*100:.2f}%")
    
    # Collect incorrectly predicted samples for these classes
    incorrect_samples = {class_id: [] for class_id in low_accuracy_classes.keys()}
    
    model.eval()
    print("\nStarting to collect incorrectly predicted samples...")
    with torch.no_grad():
        # Use tqdm to create progress bar, set leave=True to keep progress bar
        for inputs, labels in tqdm(val_loader, desc="Collecting incorrect predictions", leave=True, ncols=100):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # Handle case where outputs is a tuple, take first element as logits
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            _, preds = torch.max(outputs, 1)
            
            # Find incorrectly predicted samples
            for i, (label, pred) in enumerate(zip(labels, preds)):
                label_item = label.item()
                pred_item = pred.item()
                
                # If it's a low accuracy class and prediction is incorrect
                if label_item in low_accuracy_classes and label_item != pred_item:
                    # Only save specified number of samples
                    if len(incorrect_samples[label_item]) < max_per_class:
                        incorrect_samples[label_item].append({
                            'image': inputs[i].cpu(),
                            'true_label': label_item,
                            'pred_label': pred_item
                        })
    
    # Ensure visualization directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate heatmaps for each low accuracy class
    total_generated = 0
    
    for class_id, samples in incorrect_samples.items():
        class_name = id_to_name[class_id]
        accuracy = low_accuracy_classes[class_id]
        
        print(f"\nGenerating heatmaps for class {class_name} (accuracy: {accuracy*100:.2f}%)")
        
        # Limit number of heatmaps per class
        for i, sample in enumerate(samples):
            # Limit total number of heatmaps
            if total_generated >= max_total:
                print(f"Reached maximum heatmap limit ({max_total})")
                break
                
            try:
                # Generate Grad-CAM
                with torch.enable_grad():
                    cam = model.grad_cam.generate_cam(sample['image'].unsqueeze(0).to(device), sample['true_label'])
                
                # Process image
                image = sample['image'].numpy().transpose(1, 2, 0)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                overlayed_image = model.grad_cam.overlay_cam(image, cam)
                
                # Build safe filename
                safe_class_name = get_class_basename(class_name).replace('\\', '_').replace('/', '_')
                save_path = os.path.join(save_dir, f'grad_cam_{safe_class_name}_acc{accuracy*100:.1f}_sample{i+1}.png')
                
                # Get predicted class name
                pred_class_name = get_class_basename(id_to_name[sample['pred_label']])
                
                # Save heatmap
                plt.figure(figsize=(10, 10))
                plt.imshow(overlayed_image)
                
                # Get last part of true class name
                true_class_display = get_class_basename(class_name)
                
                plt.title(f'True Class: {true_class_display}\nPredicted Class: {pred_class_name}\nClass Accuracy: {accuracy*100:.2f}%')
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()  # Immediately close image to release memory
                
                print(f"Saved heatmap: {save_path}")
                total_generated += 1
                
            except Exception as e:
                print(f"Error generating heatmap for class {class_name}: {str(e)}")
                continue
            
            # Clear memory every 5 images
            if total_generated % 5 == 0:
                torch.cuda.empty_cache()
    
    print(f"\nHeatmap generation completed, generated {total_generated} heatmaps in total")

def visualize_confusion_matrix(confusion_matrix, class_names, epoch, save_dir='visualization', use_chinese=False):
    """Visualize confusion matrix"""
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create confusion matrix plot
    plt.figure(figsize=(20, 16))
    
    # Get base names of classes
    display_names = [get_class_basename(name) for name in class_names]
    
    # Create heatmap
    sns.heatmap(
        confusion_matrix, 
        annot=False,  # Don't show values, too many classes would overlap
        cmap='Blues',
        fmt='d', 
        xticklabels=display_names,
        yticklabels=display_names
    )
    
    # Choose title based on whether Chinese is supported
    if use_chinese:
        plt.title(f'Epoch {epoch+1} Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('True Class', fontsize=14)
    else:
        plt.title(f'Confusion Matrix - Epoch {epoch+1}', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('True Class', fontsize=14)
    
    # Adjust label size and rotation angle
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Save image
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png'), dpi=300)
    plt.close()

def analyze_performance(val_results, class_mapping, epoch, save_dir='visualization', use_chinese=False):
    """Analyze model performance and generate performance report"""
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create reverse mapping (id to name)
    id_to_name = {v: get_class_basename(k) for k, v in class_mapping.items()}
    
    # Get performance metrics for each class
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
    
    # Sort by F1 score
    sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # Create performance report
    report_file = os.path.join(save_dir, f'performance_report_epoch_{epoch+1}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        # Write overall performance
        if use_chinese:
            f.write(f"Model Performance Report - Epoch {epoch+1}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall Accuracy: {val_results['accuracy']*100:.2f}%\n")
            f.write(f"Macro F1: {val_results['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {val_results['weighted_f1']:.4f}\n\n")
            f.write("Class-wise Metrics:\n")
            f.write("-"*50 + "\n")
        else:
            f.write(f"Model Performance Report - Epoch {epoch+1}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall Accuracy: {val_results['accuracy']*100:.2f}%\n")
            f.write(f"Macro F1: {val_results['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {val_results['weighted_f1']:.4f}\n\n")
            f.write("Class-wise Metrics:\n")
            f.write("-"*50 + "\n")
        
        # Write performance for each class
        for class_name, metrics in sorted_classes:
            if use_chinese:
                f.write(f"Class: {class_name}\n")
                f.write(f"  Accuracy: {metrics['accuracy']*100:.2f}%\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
            else:
                f.write(f"Class: {class_name}\n")
                f.write(f"  Accuracy: {metrics['accuracy']*100:.2f}%\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
            f.write("-"*50 + "\n")
        
        # Write best and worst performing classes
        best_classes = sorted_classes[:5]
        worst_classes = sorted_classes[-5:]
        
        if use_chinese:
            f.write("\nTop 5 Best Performing Classes:\n")
        else:
            f.write("\nTop 5 Best Performing Classes:\n")
        f.write("-"*50 + "\n")
        for i, (class_name, metrics) in enumerate(best_classes):
            if use_chinese:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%\n")
            else:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%\n")
        
        if use_chinese:
            f.write("\nTop 5 Worst Performing Classes:\n")
        else:
            f.write("\nTop 5 Worst Performing Classes:\n")
        f.write("-"*50 + "\n")
        for i, (class_name, metrics) in enumerate(reversed(worst_classes)):
            if use_chinese:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%\n")
            else:
                f.write(f"{i+1}. {class_name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%\n")
    
    # Visualize worst performing classes
    plt.figure(figsize=(12, 8))
    worst_class_names = [name for name, _ in reversed(worst_classes)]
    worst_class_f1 = [metrics['f1'] for _, metrics in reversed(worst_classes)]
    
    plt.bar(range(len(worst_class_names)), worst_class_f1)
    plt.xticks(range(len(worst_class_names)), worst_class_names, rotation=45, ha='right')
    
    if use_chinese:
        plt.title(f'F1 Scores of 5 Worst Performing Classes - Epoch {epoch+1}')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
    else:
        plt.title(f'F1 Scores of 5 Worst Performing Classes - Epoch {epoch+1}')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'worst_classes_epoch_{epoch+1}.png'))
    plt.close()
    
    # Visualize best performing classes
    plt.figure(figsize=(12, 8))
    best_class_names = [name for name, _ in best_classes]
    best_class_f1 = [metrics['f1'] for _, metrics in best_classes]
    
    plt.bar(range(len(best_class_names)), best_class_f1)
    plt.xticks(range(len(best_class_names)), best_class_names, rotation=45, ha='right')
    
    if use_chinese:
        plt.title(f'F1 Scores of 5 Best Performing Classes - Epoch {epoch+1}')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
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
    """Analyze confusion matrix to find most confused class pairs"""
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create reverse mapping (id to name)
    id_to_name = {v: get_class_basename(k) for k, v in class_mapping.items()}
    
    # Find elements with largest values in off-diagonal elements (most confused class pairs)
    num_classes = confusion_matrix.shape[0]
    confusion_pairs = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > 0:
                # Calculate confusion rate (number of misclassified samples / total samples for this class)
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
    
    # Sort by confusion count
    confusion_pairs_by_count = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)[:top_n]
    
    # Sort by error rate
    confusion_pairs_by_rate = sorted(confusion_pairs, key=lambda x: x['error_rate'], reverse=True)[:top_n]
    
    # Generate report
    report_file = os.path.join(save_dir, 'confusion_pairs_analysis.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        if use_chinese:
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
    
    # Visualize confusion pairs
    plt.figure(figsize=(14, 8))
    
    # Plot bar chart of top 10 confusion pairs
    top_10_pairs = confusion_pairs_by_count[:10]
    pair_labels = [f"{p['true_class_name']}->{p['pred_class_name']}" for p in top_10_pairs]
    pair_counts = [p['count'] for p in top_10_pairs]
    
    plt.bar(range(len(pair_labels)), pair_counts)
    plt.xticks(range(len(pair_labels)), pair_labels, rotation=45, ha='right')
    
    if use_chinese:
        plt.title('Top 10 Most Confused Class Pairs')
        plt.xlabel('Class Pairs')
        plt.ylabel('Confusion Count')
    else:
        plt.title('Top 10 Most Confused Class Pairs')
        plt.xlabel('Class Pairs')
        plt.ylabel('Confusion Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_pairs.png'))
    plt.close()
    
    return report_file

def get_class_basename(class_name):
    """Extract base name from class path, handling different path separators"""
    if '\\' in class_name:
        return class_name.split('\\')[-1]
    elif '/' in class_name:
        return class_name.split('/')[-1]
    else:
        return class_name

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Rock classification model training and evaluation')
    
    # Configuration file support
    parser.add_argument('--config', type=str, default=None, 
                        help='JSON configuration file path, will override command line arguments')
    
    # Model selection parameters
    parser.add_argument('--model_type', type=str, default='ensemble',
                       choices=['ensemble', 'resnet50', 'resnet50_optimized', 'efficientnet_b4', 'inceptionv3'],
                       help='Select model type to train')
    parser.add_argument('--no_attention', action='store_true',
                       help='Set this flag to disable attention mechanism')
    
    # Learning rate scheduler parameters
    parser.add_argument('--scheduler_type', type=str, default='multistage',
                       choices=['multistage', 'cosine'],
                       help='Select learning rate scheduler type, multistage for multi-stage, cosine for single cosine')
    parser.add_argument('--cosine_T_max', type=int, default=None,
                       help='T_max parameter for cosine scheduler, defaults to total epochs')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6,
                       help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--cosine_warmup_epochs', type=int, default=10,
                       help='Warmup epochs for cosine scheduler')
    
    # Training parameters
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience value')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    
    # Evaluation parameters
    parser.add_argument('--accuracy_threshold', type=float, default=0.8, 
                        help='Accuracy threshold for heatmap generation, classes below this threshold will generate heatmaps')
    parser.add_argument('--max_per_class', type=int, default=10, 
                        help='Maximum number of heatmaps to generate per class')
    parser.add_argument('--max_total', type=int, default=50, 
                        help='Maximum total number of heatmaps to generate')
    
    # Optimized ResNet50 specific parameters
    parser.add_argument('--mining_ratio', type=float, default=0.25,
                       help='Hard example mining ratio (resnet50_optimized only)')
    parser.add_argument('--triangular_margin', type=float, default=0.8,
                       help='Triangular loss margin (resnet50_optimized only)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Other parameters
    parser.add_argument('--eval_only', action='store_true', help='Only perform evaluation, do not train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for evaluation')
    
    return parser.parse_args()

class MultiStageScheduler:
    """Performance-adaptive multi-stage learning rate scheduler
    Dynamically adjusts learning rate strategy based on model performance, rather than fixed epochs
    """
    def __init__(self, optimizer, num_epochs, init_lr, min_lr):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # Minimum epochs requirement for each stage - increased minimum epochs requirement
        self.min_epochs_per_stage = {
            'warmup': max(5, int(num_epochs * 0.05)),  # At least 5 epochs or 5% of total epochs
            'aggressive': max(15, int(num_epochs * 0.15)),  # At least 15 epochs or 15% of total epochs
            'refinement': max(20, int(num_epochs * 0.2)),  # At least 20 epochs or 20% of total epochs
            'fine_tuning': max(10, int(num_epochs * 0.1))   # At least 10 epochs or 10% of total epochs
        }
        
        # Current stage
        self.current_stage = 'warmup'
        self.current_stage_epochs = 0
        
        # Track performance and learning rate changes
        self.accuracy_history = []
        self.lr_history = []
        self.best_accuracy = 0
        self.plateau_counter = 0
        
        # Stage switching conditions - increased tolerance
        self.stagnation_threshold = 5  # Performance stagnation epoch threshold increased to 5
        self.improvement_threshold = 0.002  # Performance improvement threshold lowered, easier to detect progress
        
        # Schedulers used for each stage
        self.schedulers = {}
        self._setup_schedulers()
        
        # Record learning rate at end of previous stage
        self.last_stage_lr = init_lr
        
        # Set up learning rate protection mechanism
        self.lr_protection = {
            'max_decrease_factor': 0.5,  # Maximum decrease ratio per step
            'min_lr_factor': 0.1,  # Minimum ratio relative to initial learning rate
            'recovery_factor': 1.2  # Recovery coefficient when performance improves
        }
        
        # Record boundaries of each stage for visualization
        self.stage_boundaries = {
            'warmup_end': 0,
            'aggressive_end': 0,
            'refinement_end': 0
        }
        
        # Smooth transition mechanism
        self.transition_steps = 3  # Number of smooth steps for stage transition
        self.in_transition = False
        self.transition_from = None
        self.transition_to = None
        self.transition_step = 0
        
    def _setup_schedulers(self):
        """Set up learning rate schedulers for each stage"""
        # Warmup stage: linear increase to initial learning rate, smoother
        self.schedulers['warmup'] = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.2,  # Increased starting factor
            end_factor=1.0,
            total_iters=self.min_epochs_per_stage['warmup']
        )
        
        # Aggressive exploration stage: 1cycle policy, smoother parameters
        self.schedulers['aggressive'] = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.init_lr * 1.5,  # Reduced maximum learning rate to prevent excessive fluctuations
            total_steps=self.min_epochs_per_stage['aggressive'] * 2,
            pct_start=0.4,  # Increased rise proportion
            anneal_strategy='cos',
            div_factor=5.0,  # Reduced initial decrease magnitude
            final_div_factor=4.0  # Reduced final decrease magnitude
        )
        
        # Refinement stage: cosine annealing with larger period and higher minimum
        self.schedulers['refinement'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.min_epochs_per_stage['refinement'] * 3,  # Increased period length
            eta_min=self.min_lr * 20  # Increased minimum learning rate
        )
        
    def step(self, accuracy=None):
        """Update learning rate, dynamically adjust stage based on current performance"""
        # Update current epoch
        self.current_epoch += 1
        self.current_stage_epochs += 1
        
        # Record accuracy history
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
            
            # Update best accuracy and stagnation counter
            if accuracy > self.best_accuracy + 0.0005:  # Added small tolerance
                self.best_accuracy = accuracy
                self.plateau_counter = 0
            else:
                self.plateau_counter += 1
                
        # If in stage transition, handle smooth transition
        if self.in_transition:
            result = self._handle_transition()
            if result:  # If transition completed
                self.in_transition = False
                self.current_stage = self.transition_to
                self.transition_to = None
                self.transition_from = None
                self.transition_step = 0
                self.current_stage_epochs = 0
                self.plateau_counter = 0
        # Check if should start new stage transition
        elif self._should_switch_stage():
            next_stage = self._get_next_stage()
            if next_stage != self.current_stage:
                # Record current stage end epoch for visualization
                if self.current_stage == 'warmup':
                    self.stage_boundaries['warmup_end'] = self.current_epoch
                elif self.current_stage == 'aggressive':
                    self.stage_boundaries['aggressive_end'] = self.current_epoch
                elif self.current_stage == 'refinement':
                    self.stage_boundaries['refinement_end'] = self.current_epoch
                
                # Start stage transition
                self.in_transition = True
                self.transition_from = self.current_stage
                self.transition_to = next_stage
                self.transition_step = 0
        
        # Update learning rate based on current stage
        if not self.in_transition:
            if self.current_stage in ['warmup', 'aggressive', 'refinement']:
                # Use predefined scheduler
                self.schedulers[self.current_stage].step()
            else:
                # Fine-tuning stage: adaptively adjust learning rate based on validation accuracy
                self._adaptive_step(accuracy)
        
        # Apply learning rate protection mechanism
        self._apply_lr_protection()
        
        # Record current learning rate
        current_lr = self.get_last_lr()
        self.lr_history.append(current_lr[0])
        
        return current_lr
    
    def _handle_transition(self):
        """Handle smooth transition between stages"""
        self.transition_step += 1
        
        # Get learning rates of source and target stages
        if self.transition_from in ['warmup', 'aggressive', 'refinement']:
            from_lr = self.schedulers[self.transition_from].get_last_lr()[0]
        else:
            from_lr = self.get_last_lr()[0]
            
        # Calculate target learning rate
        if self.transition_to == 'aggressive':
            # Aggressive stage starting learning rate should be close to end of previous stage
            to_lr = from_lr * 1.1  # Slight increase to start exploration
        elif self.transition_to == 'refinement':
            # Refinement stage should start from current learning rate
            to_lr = from_lr
        elif self.transition_to == 'fine_tuning':
            # Fine-tuning stage should use lower but not too low learning rate
            to_lr = max(from_lr * 0.7, self.min_lr * 30)
        else:
            to_lr = from_lr
            
        # Calculate current learning rate using cosine transition
        progress = self.transition_step / self.transition_steps
        transition_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        current_lr = to_lr + (from_lr - to_lr) * transition_factor
        
        # Apply transition learning rate
        for group in self.optimizer.param_groups:
            group['lr'] = current_lr
            
        # If transition completed, initialize next stage
        if self.transition_step >= self.transition_steps:
            self._initialize_next_stage(self.transition_to, current_lr)
            return True
            
        return False
            
    def _initialize_next_stage(self, new_stage, current_lr):
        """Initialize scheduler for new stage"""
        self.last_stage_lr = current_lr
        
        # Reinitialize scheduler based on new stage
        if new_stage == 'aggressive':
            self.schedulers['aggressive'] = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=current_lr * 1.5,  # Maximum relative to current learning rate
                total_steps=self.min_epochs_per_stage['aggressive'] * 2,
                pct_start=0.4,
                anneal_strategy='cos',
                div_factor=3.0,  # Reduce fluctuations
                final_div_factor=3.0
            )
        elif new_stage == 'refinement':
            self.schedulers['refinement'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.min_epochs_per_stage['refinement'] * 3,
                eta_min=max(self.min_lr * 20, current_lr * 0.2)  # Ensure minimum is not too low
            )
    
    def _should_switch_stage(self):
        """Check if should switch to next stage"""
        # Not enough accuracy history data
        if len(self.accuracy_history) < 5:  # Increased to at least 5 samples
            return False
            
        # Check if minimum epochs requirement is met
        if self.current_stage_epochs < self.min_epochs_per_stage[self.current_stage]:
            return False
            
        # Set different switching strategies based on current stage
        if self.current_stage == 'warmup':
            # Warmup stage: switch when average accuracy improvement in recent 5 epochs is below threshold
            if len(self.accuracy_history) < 7:
                return False
                
            recent_improvements = [self.accuracy_history[i] - self.accuracy_history[i-1] 
                                 for i in range(len(self.accuracy_history)-5, len(self.accuracy_history))]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            return avg_improvement < 0.01 and avg_improvement > 0  # Growth slowed but still positive
            
        elif self.current_stage == 'aggressive':
            # Aggressive exploration stage: switch when accuracy stabilizes or reaches high level
            recent_improvements = [self.accuracy_history[i] - self.accuracy_history[i-1] 
                                 for i in range(len(self.accuracy_history)-4, len(self.accuracy_history))]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            
            # Switch when accuracy growth slows or has reached high level
            high_accuracy_threshold = 0.85  # High accuracy threshold
            return (avg_improvement < self.improvement_threshold or 
                   (self.accuracy_history[-1] > high_accuracy_threshold and self.current_stage_epochs > self.min_epochs_per_stage['aggressive'] * 1.5))
            
        elif self.current_stage == 'refinement':
            # Refinement stage: switch when long stagnation occurs or approaching max epochs
            max_epochs_factor = 1.5  # Maximum epochs factor
            approaching_max_epochs = self.current_stage_epochs > self.min_epochs_per_stage['refinement'] * max_epochs_factor
            return self.plateau_counter >= self.stagnation_threshold or approaching_max_epochs
            
        return False
    
    def _get_next_stage(self):
        """Get next stage"""
        if self.current_stage == 'warmup':
            return 'aggressive'
        elif self.current_stage == 'aggressive':
            return 'refinement'
        elif self.current_stage == 'refinement':
            return 'fine_tuning'
        return 'fine_tuning'
    
    def _transition_to_stage(self, new_stage):
        """Handle learning rate transition during stage switching"""
        # This method has been replaced by _handle_transition and _initialize_next_stage
        pass
    
    def _adaptive_step(self, accuracy):
        """Adaptive learning rate adjustment for fine-tuning stage"""
        if accuracy is None:
            return
        
        # Track accuracy history to detect trends
        window_size = min(5, len(self.accuracy_history))
        if window_size < 3:
            return
            
        recent_accuracies = self.accuracy_history[-window_size:]
        
        # Calculate trend - linear regression slope
        x = list(range(window_size))
        slope = sum((x[i] - sum(x)/window_size) * (recent_accuracies[i] - sum(recent_accuracies)/window_size) 
                  for i in range(window_size)) / sum((x[i] - sum(x)/window_size)**2 for i in range(window_size))
        
        current_lr = self.get_last_lr()[0]
        
        # Adjust learning rate based on trend
        if slope < -0.001:  # Clear downward trend
            # Increase learning rate to escape local minimum
            new_lr = min(current_lr * 2.0, self.init_lr * 0.1)
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr
            self.plateau_counter = 0
            
        elif abs(slope) < 0.0005:  # Plateau period
            if self.plateau_counter >= 3:  # Sustained plateau
                # Gently decrease learning rate
                new_lr = max(current_lr * 0.8, self.min_lr)
                for group in self.optimizer.param_groups:
                    group['lr'] = new_lr
                self.plateau_counter = 0
            
        else:  # Upward trend, maintain current learning rate
            pass
    
    def _apply_lr_protection(self):
        """Apply learning rate protection mechanism to prevent sudden drops or excessively low learning rates"""
        if not self.lr_history:
            return
            
        previous_lr = self.lr_history[-1] if len(self.lr_history) > 0 else self.init_lr
        current_lr = self.get_last_lr()[0]
        
        # Prevent excessive single-step decrease
        if current_lr < previous_lr * self.lr_protection['max_decrease_factor']:
            protected_lr = previous_lr * self.lr_protection['max_decrease_factor']
            for group in self.optimizer.param_groups:
                group['lr'] = protected_lr
                
        # Ensure learning rate doesn't fall below minimum protection value
        min_protected_lr = self.init_lr * self.lr_protection['min_lr_factor']
        if current_lr < min_protected_lr and self.current_stage != 'fine_tuning':
            for group in self.optimizer.param_groups:
                group['lr'] = min_protected_lr
                
        # If performance improves, give learning rate recovery opportunity
        if len(self.accuracy_history) >= 2 and self.accuracy_history[-1] > self.accuracy_history[-2] + 0.005:
            current_lr = self.get_last_lr()[0]  # Get current learning rate that may have been protected
            if current_lr < previous_lr and self.current_stage == 'fine_tuning':
                recovery_lr = min(current_lr * self.lr_protection['recovery_factor'], previous_lr)
                for group in self.optimizer.param_groups:
                    group['lr'] = recovery_lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_stage_info(self):
        """Get current stage information for logging"""
        stage = self.transition_to if self.in_transition else self.current_stage
        stage_display = f"{self.transition_from}->{stage} (Transition {self.transition_step}/{self.transition_steps})" if self.in_transition else stage
        
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
    # Parse command line arguments
    args = parse_args()
    
    # Set up matplotlib, check if Chinese is supported
    use_chinese = setup_matplotlib()
    
    # Initialize configuration parameters
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
        'use_chinese': use_chinese,  # Add Chinese support flag
        'eval_frequency': 20,  # Evaluate every 20 epochs
        'model_type': args.model_type,  # Model type
        'use_attention': not args.no_attention,  # Whether to use attention mechanism
        'scheduler_type': args.scheduler_type,  # Learning rate scheduler type
        'cosine_T_max': args.cosine_T_max if args.cosine_T_max else args.num_epochs,  # T_max parameter for cosine scheduler
        'cosine_eta_min': args.cosine_eta_min,  # Minimum learning rate for cosine scheduler
        'cosine_warmup_epochs': args.cosine_warmup_epochs,  # Warmup epochs for cosine scheduler
        # Optimized ResNet50 specific parameters
        'mining_ratio': args.mining_ratio,  # Hard example mining ratio
        'triangular_margin': args.triangular_margin,  # Triangular loss margin
        'gradient_clip': args.gradient_clip  # Gradient clipping threshold
    }
    
    # If using optimized ResNet50, automatically adjust some parameters
    if config['model_type'] == 'resnet50_optimized':
        print("Detected optimized ResNet50, automatically applying optimized configuration...")
        
        # If batch_size is large, automatically reduce to accommodate more complex model
        if config['batch_size'] > 16:
            original_batch_size = config['batch_size']
            config['batch_size'] = max(8, config['batch_size'] // 2)
            print(f"Automatically adjusted batch_size: {original_batch_size} -> {config['batch_size']}")
        
        # If learning rate is high, automatically reduce
        if config['learning_rate'] > 5e-5:
            original_lr = config['learning_rate']
            config['learning_rate'] = config['learning_rate'] * 0.8
            print(f"Automatically adjusted learning rate: {original_lr} -> {config['learning_rate']}")
        
        # Increase weight decay
        original_weight_decay = config['weight_decay']
        config['weight_decay'] = config['weight_decay'] * 1.2
        print(f"Automatically adjusted weight decay: {original_weight_decay} -> {config['weight_decay']}")
        
        # Increase patience value
        config['patience'] = config['patience'] + 5
        print(f"Increased early stopping patience to: {config['patience']}")
        
        # Ensure gradient accumulation steps is at least 2
        if config['accumulation_steps'] < 2:
            config['accumulation_steps'] = 2
            print(f"Set gradient accumulation steps to: {config['accumulation_steps']}")
    
    # If configuration file is provided, load parameters from it
    if args.config and os.path.exists(args.config):
        try:
            print(f"Loading parameters from configuration file: {args.config}")
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                # Update configuration parameters
                config.update(file_config)
                print(f"Successfully loaded configuration file")
        except Exception as e:
            print(f"Error loading configuration file: {str(e)}")
    
    # Create save directories
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
    
    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # If specific GPU is specified, use the specified GPU
    if 'cuda_device' in config and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['cuda_device']}")
    
    # Optimize CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
    
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set up logging
    logger = setup_logger(log_dir)
    logger.info(f'Using device: {device}')
    logger.info('Configuration parameters:')
    for k, v in config.items():
        logger.info(f'{k}: {v}')
    
    try:
        # Load data and class mapping
        with open(os.path.join(config['data_dir'], 'class_mapping.json'), 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
            num_classes = len(class_mapping)
            # Create reverse mapping (id to name)
            id_to_name = {v: k for k, v in class_mapping.items()}
        
        train_loader, val_loader = create_data_loaders(
            config['data_dir'],
            config['batch_size'],
            num_workers=config.get('num_workers', min(8, os.cpu_count()))
        )
        
        # Create model and apply configuration parameters
        if config['model_type'] == 'ensemble':
            model = EnsembleModel(
                num_classes=num_classes,
                temperature=config.get('temperature', 2.0)
            ).to(device)
        else:
            # Use our newly created model functions
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
        
        # Output selected model information
        logger.info(f"Using model: {config['model_type']}, Attention mechanism: {config['use_attention']}")
        print(f"Using model: {config['model_type']}, Attention mechanism: {config['use_attention']}")
        
        # If model parameters are configured, apply them
        if 'dropout_rate' in config:
            # Update Dropout rate
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = config['dropout_rate']
        
        if 'inception_freeze_layers' in config:
            # Update Inception model frozen layer count
            freeze_layers = int(config['inception_freeze_layers'])
            for param in list(model.inception.inception.parameters())[:-freeze_layers]:
                param.requires_grad = False
        
        if 'fpn_channels' in config:
            # Update FPN channel count
            fpn_channels = int(config['fpn_channels'])
            model.inception.fpn.out_channels = fpn_channels
            model.efficientnet.fpn.out_channels = fpn_channels
        
        # Update loss function parameters
        if hasattr(model, 'combined_loss') and hasattr(model.combined_loss, 'focal_loss'):
            if 'focal_gamma' in config:
                model.combined_loss.focal_loss.gamma = config['focal_gamma']
            if 'label_smoothing' in config:
                model.combined_loss.focal_loss.smoothing = config['label_smoothing']
        
        if hasattr(model, 'kd_loss') and 'temperature' in config:
            model.kd_loss.temperature = config['temperature']
        
        # If only performing evaluation
        if args.eval_only:
            if args.checkpoint:
                checkpoint_path = args.checkpoint
            else:
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                
            if os.path.exists(checkpoint_path):
                print(f"\nLoading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                # Use strict=False to allow loading partial weights, ignore mismatched keys
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Model loaded successfully, some layers may not be loaded (model structure may have been updated)")
                
                # Perform evaluation
                final_results = validate(model, val_loader, device)
                
                # Output evaluation results
                print("\nEvaluation results:")
                print(f"Accuracy: {final_results['accuracy']*100:.2f}%")
                print(f"Macro F1: {final_results['macro_f1']:.4f}")
                print(f"Weighted F1: {final_results['weighted_f1']:.4f}")
                
                # Generate confusion matrix
                visualize_confusion_matrix(
                    final_results['confusion_matrix'],
                    list(class_mapping.keys()),
                    epoch=999,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                
                # Generate performance report
                final_report_file = analyze_performance(
                    final_results,
                    class_mapping,
                    epoch=999,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                print(f"Performance report saved to: {final_report_file}")
                
                # Generate heatmaps
                print("\nStarting to generate heatmaps for low accuracy classes...")
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
                
                # Analyze confusion matrix to find most confused class pairs
                print("\nStarting to analyze confusion matrix to find most confused class pairs...")
                confusion_pairs_file = analyze_confusion_pairs(
                    final_results['confusion_matrix'],
                    class_mapping,
                    top_n=20,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                print(f"Confusion matrix analysis saved to: {confusion_pairs_file}")
                
                return
            else:
                print(f"Error: Checkpoint file does not exist: {checkpoint_path}")
                return
        
        # Optimizer - use configuration parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('eps', 1e-8)
        )
        
        # Learning rate scheduler selection
        if config['scheduler_type'] == 'multistage':
            # Use multi-stage learning rate scheduler
            scheduler = MultiStageScheduler(
                optimizer,
                config['num_epochs'],
                config['learning_rate'],
                config.get('min_lr', 1e-6)
            )
            logger.info(f"Using multi-stage learning rate scheduler")
            print(f"Using multi-stage learning rate scheduler")
        else:
            # Use single cosine learning rate scheduler with warmup
            if config['cosine_warmup_epochs'] > 0:
                # Linear warmup + cosine annealing
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
                
                # Sequential scheduler
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[config['cosine_warmup_epochs']]
                )
            else:
                # Use only cosine annealing
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config['cosine_T_max'],
                    eta_min=config['cosine_eta_min']
                )
            
            logger.info(f"Using single cosine learning rate scheduler, T_max={config['cosine_T_max']}, warmup_epochs={config['cosine_warmup_epochs']}")
            print(f"Using single cosine learning rate scheduler, T_max={config['cosine_T_max']}, warmup_epochs={config['cosine_warmup_epochs']}")
        
        # Use mixed precision training
        scaler = GradScaler(
            init_scale=config.get('init_scale', 2**10),
            growth_factor=config.get('growth_factor', 2.0),
            backoff_factor=config.get('backoff_factor', 0.5),
            growth_interval=config.get('growth_interval', 100)
        )
        
        # Early stopping mechanism
        early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Loss monitoring
        loss_monitor = LossMonitor()
        
        # Training loop
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
        
        # Dictionary to store current class accuracies
        current_class_accuracies = None
        
        for epoch in range(config['num_epochs']):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}]")
            print(f"\n{'='*50}")
            print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
            
            # Set data augmentation parameters (ensure relevant transforms exist)
            try:
                # Apply MixUp and RandomErasing parameters
                if hasattr(train_loader.dataset, 'transform'):
                    for transform in train_loader.dataset.transform.transforms:
                        # Update MixUp alpha
                        if hasattr(transform, 'alpha') and 'mixup_alpha' in config:
                            transform.alpha = config['mixup_alpha']
                        # Update RandomErasing probability
                        if hasattr(transform, 'p') and hasattr(transform, 'value') and 'random_erasing_prob' in config:
                            transform.p = config['random_erasing_prob']
            except Exception as e:
                logger.warning(f"Failed to set data augmentation parameters: {str(e)}")
            
            # Training, pass current class accuracies
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer,
                scaler, device, epoch, config['num_epochs'],
                config.get('gradient_clip', 1.0),
                class_accuracies=current_class_accuracies,
                accumulation_steps=config.get('accumulation_steps', 1)
            )
            
            # Validation
            val_results = validate(model, val_loader, device)
            
            # Update current class accuracies
            current_class_accuracies = val_results['class_accuracies']
            
            # Update learning rate
            if config['scheduler_type'] == 'multistage':
                current_lr = scheduler.step(val_results['accuracy'])
                stage_info = scheduler.get_stage_info()
            else:
                scheduler.step()
                current_lr = [group['lr'] for group in optimizer.param_groups]
                stage_info = {'stage': 'cosine', 'epoch_in_stage': epoch, 'plateau_counter': 0}
            
            # Update training history
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_acc'].append(val_results['accuracy'])
            train_history['val_loss'].append(val_results['loss'])
            train_history['learning_rates'].append(current_lr[0])
            train_history['macro_f1'].append(val_results['macro_f1'])
            train_history['weighted_f1'].append(val_results['weighted_f1'])
            
            # Add stage information to training history
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
            
            # Update evaluation metrics history for each class
            for class_id, accuracy in val_results['class_accuracies'].items():
                class_name = id_to_name[class_id]
                
                # Update accuracy history
                if class_name not in train_history['class_acc']:
                    train_history['class_acc'][class_name] = []
                train_history['class_acc'][class_name].append(accuracy)
                
                # Update F1 score history
                if class_name not in train_history['class_f1']:
                    train_history['class_f1'][class_name] = []
                train_history['class_f1'][class_name].append(val_results['class_f1'][class_id])
                
                # Update precision history
                if class_name not in train_history['class_precision']:
                    train_history['class_precision'][class_name] = []
                train_history['class_precision'][class_name].append(val_results['class_precision'][class_id])
                
                # Update recall history
                if class_name not in train_history['class_recall']:
                    train_history['class_recall'][class_name] = []
                train_history['class_recall'][class_name].append(val_results['class_recall'][class_id])
            
            # Output training information
            print(f"Training accuracy: {train_acc*100:.2f}%")
            print(f"Validation accuracy: {val_results['accuracy']*100:.2f}%")
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {val_results['loss']:.4f}")
            print(f"Macro F1: {val_results['macro_f1']:.4f}")
            print(f"Weighted F1: {val_results['weighted_f1']:.4f}")
            print(f"Current learning rate: {current_lr[0]:.6f} (Stage: {stage_info['stage']}, Epoch {stage_info['epoch_in_stage']} in stage, Plateau counter: {stage_info['plateau_counter']})")
            print(f"{'='*50}\n")
            
            # Log to logger
            logger.info(f"\nTraining statistics:")
            logger.info(f"Learning rate: {current_lr[0]:.6f} (Stage: {stage_info['stage']}, Epoch {stage_info['epoch_in_stage']} in stage)")
            logger.info(f"Training loss: {train_loss:.4f}")
            logger.info(f"Validation loss: {val_results['loss']:.4f}")
            logger.info(f"Training accuracy: {train_acc*100:.2f}%")
            logger.info(f"Validation accuracy: {val_results['accuracy']*100:.2f}%")
            logger.info(f"Macro F1: {val_results['macro_f1']:.4f}")
            logger.info(f"Weighted F1: {val_results['weighted_f1']:.4f}")
            
            # Only generate confusion matrix and performance analysis at specified frequency
            if epoch % config['eval_frequency'] == 0 or epoch == config['num_epochs'] - 1:
                # Visualize confusion matrix
                visualize_confusion_matrix(
                    val_results['confusion_matrix'], 
                    list(class_mapping.keys()), 
                    epoch,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                
                # Analyze performance
                performance_report_file = analyze_performance(
                    val_results, 
                    class_mapping, 
                    epoch,
                    save_dir=visualization_dir,
                    use_chinese=config['use_chinese']
                )
                logger.info(f'Performance report saved to: {performance_report_file}')
            
            # Save best model
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                                
                # Save model
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
                logger.info("Best model saved")
            
            # Check early stopping
            early_stopping(train_loss, val_results['accuracy'], epoch)
            if early_stopping.early_stop:
                print(f"\n[Early Stopping] Early stopping triggered, stopping training")
                print(f"Best validation accuracy: {best_val_acc*100:.2f}% (Epoch {early_stopping.best_epoch + 1})")
                print(f"Current epoch: {epoch + 1}")
                print(f"No improvement for {config['patience']} epochs, stopping training")
                logger.info('\nEarly stopping triggered, stopping training')
                logger.info(f"Best validation accuracy: {best_val_acc*100:.2f}% (Epoch {early_stopping.best_epoch + 1})")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc*100:.2f}%")
        
        # Save training history
        history_file = os.path.join(log_dir, 'ensemble_training_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(train_history, f, ensure_ascii=False, indent=4)
        logger.info(f'Training history saved to: {history_file}')
        
        # Plot learning curves
        plot_learning_curves(
            train_history, 
            save_dir=visualization_dir,
            use_chinese=config['use_chinese']
        )
        logger.info('Learning curves generated')
        
        # Load best model for final evaluation
        print("\nLoading best model for final evaluation...")
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, weights_only=False)
            # Use strict=False to allow loading partial weights, ignore mismatched keys
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Model loaded successfully, some layers may not be loaded (model structure may have been updated)")
            
            # Perform final evaluation
            final_results = validate(model, val_loader, device)
            
            # Generate final confusion matrix
            visualize_confusion_matrix(
                final_results['confusion_matrix'],
                list(class_mapping.keys()),
                epoch=999,  # Use special value to represent final results
                save_dir=visualization_dir,
                use_chinese=config['use_chinese']
            )
            
            # Generate final performance report
            final_report_file = analyze_performance(
                final_results,
                class_mapping,
                epoch=999,  # Use special value to represent final results
                save_dir=visualization_dir,
                use_chinese=config['use_chinese']
            )
            
            # Output final results
            print("\nFinal evaluation results:")
            print(f"Accuracy: {final_results['accuracy']*100:.2f}%")
            print(f"Macro F1: {final_results['macro_f1']:.4f}")
            print(f"Weighted F1: {final_results['weighted_f1']:.4f}")
            print(f"Final performance report saved to: {final_report_file}")
            
            # Log to logger
            logger.info("\nFinal evaluation results:")
            logger.info(f"Accuracy: {final_results['accuracy']*100:.2f}%")
            logger.info(f"Macro F1: {final_results['macro_f1']:.4f}")
            logger.info(f"Weighted F1: {final_results['weighted_f1']:.4f}")
            
            # Generate heatmaps for incorrectly predicted images of low accuracy classes
            print("\nStarting to generate heatmaps for low accuracy classes...")
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
            print("Heatmap generation completed!")
            
            # Analyze confusion matrix to find most confused class pairs
            print("\nStarting to analyze confusion matrix to find most confused class pairs...")
            confusion_pairs_file = analyze_confusion_pairs(
                final_results['confusion_matrix'],
                class_mapping,
                top_n=20,
                save_dir=visualization_dir,
                use_chinese=config['use_chinese']
            )
            logger.info(f'Confusion matrix analysis saved to: {confusion_pairs_file}')
        else:
            logger.warning(f"Best model file not found: {best_model_path}")
            
    except Exception as e:
        logger.error(f'Error occurred during training: {str(e)}')
        raise e

if __name__ == "__main__":
    main() 