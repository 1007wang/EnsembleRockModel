#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rock Classification Model Testing Script

Used to test trained models (ensemble and resnet50_optimized) on test datasets.
Supports evaluation of overall accuracy, Top-K accuracy, category accuracy, and confusion matrix generation.

Usage:
    python src/test_model.py --model ensemble
    python src/test_model.py --model resnet_optimized
    python src/test_model.py --model all
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.backbone_models import create_model
from src.models.ensemble_model import EnsembleModel


# ============================================================================
# Constants
# ============================================================================

# Number of classes
NUM_CLASSES = 49

# Major category definitions (index ranges)
ROCK_CATEGORIES = {
    'IgneousRocks': list(range(0, 16)),       # Igneous: 0-15
    'MetamorphicRocks': list(range(16, 29)),  # Metamorphic: 16-28
    'SedimentaryRocks': list(range(29, 49))   # Sedimentary: 29-48
}

# Major category names
CATEGORY_NAMES = {
    'IgneousRocks': 'Igneous',
    'MetamorphicRocks': 'Metamorphic',
    'SedimentaryRocks': 'Sedimentary'
}

# Default paths
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_TEST_DATA_DIR = 'test_data'
DEFAULT_OUTPUT_DIR = 'test_reports'


# ============================================================================
# Test Dataset
# ============================================================================

class TestDataset(Dataset):
    """Test dataset loader"""
    
    def __init__(self, data_dir, class_mapping_path=None, transform=None):
        """
        Initialize test dataset
        
        Args:
            data_dir: Test data directory path
            class_mapping_path: Class mapping file path (optional, default is in data_dir)
            transform: Image transformation operations
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load class mapping
        if class_mapping_path is None:
            class_mapping_path = os.path.join(data_dir, 'class_mapping.json')
        
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
        
        # Create reverse mapping (index -> class path)
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Collect all test samples
        self.samples = []
        self._collect_samples()
        
        print(f"Test dataset loaded: {len(self.samples)} samples, {len(self.class_mapping)} classes")
    
    def _collect_samples(self):
        """Collect all test image samples"""
        for class_path, class_idx in self.class_mapping.items():
            full_path = os.path.join(self.data_dir, class_path)
            if os.path.exists(full_path):
                for img_name in sorted(os.listdir(full_path)):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(full_path, img_name)
                        self.samples.append({
                            'path': img_path,
                            'label': class_idx,
                            'class_name': class_path
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        label = sample['label']
        
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                return img, label, img_path
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            # Return a default tensor
            default_tensor = torch.zeros((3, 299, 299))
            return default_tensor, label, img_path


def get_test_transform():
    """Get image transformation for testing"""
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_type, checkpoint_path, device):
    """
    Load model of specified type
    
    Args:
        model_type: Model type ('ensemble' or 'resnet_optimized')
        checkpoint_path: Checkpoint file path
        device: Running device
    
    Returns:
        Loaded model
    """
    print(f"Loading model: {model_type}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Create model
    if model_type == 'ensemble':
        model = EnsembleModel(num_classes=NUM_CLASSES)
    elif model_type == 'resnet_optimized':
        # Detect if checkpoint contains attention module weights
        has_attention = any('attention' in key.lower() for key in state_dict.keys())
        use_attention = has_attention
        model = create_model('resnet50_optimized', num_classes=NUM_CLASSES, use_attention=use_attention)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load weights (use strict=False to allow partial loading)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # Try lenient loading
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully: {model_type}")
    return model


def discover_models(checkpoint_dir):
    """
    Automatically discover available model checkpoints
    
    Args:
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Dictionary containing discovered models and their paths
    """
    models = {}
    
    # Check ensemble.pth
    ensemble_path = os.path.join(checkpoint_dir, 'ensemble.pth')
    if os.path.exists(ensemble_path):
        models['ensemble'] = ensemble_path
    
    # Check resnet50_optimized.pth
    resnet_path = os.path.join(checkpoint_dir, 'resnet50_optimized.pth')
    if os.path.exists(resnet_path):
        models['resnet_optimized'] = resnet_path
    
    return models


# ============================================================================
# Helper Functions
# ============================================================================

def get_rock_type(class_path):
    """Get rock major category based on class path"""
    for category, indices in ROCK_CATEGORIES.items():
        if class_path.startswith(category):
            return CATEGORY_NAMES[category]
    return "Unknown"


def get_rock_type_en(class_path):
    """Get rock major category (English) based on class path"""
    for category in ROCK_CATEGORIES.keys():
        if class_path.startswith(category):
            return category.replace('Rocks', '')
    return "Unknown"


def get_class_short_name(class_path):
    """Get short name of the class"""
    if '\\' in class_path:
        return class_path.split('\\')[-1]
    elif '/' in class_path:
        return class_path.split('/')[-1]
    return class_path


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, dataloader, device, class_mapping):
    """
    Evaluate model performance on test set
    
    Args:
        model: Model to evaluate
        dataloader: Test data loader
        device: Running device
        class_mapping: Class mapping
    
    Returns:
        Evaluation results dictionary
    """
    model.eval()
    
    # Create reverse mapping
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    predictions_detail = []
    
    sample_idx = 0
    total_samples = len(dataloader.dataset)
    correct_count = 0
    
    print("\n" + "=" * 70)
    print("Detailed Prediction Results")
    print("=" * 70)
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward propagation
            outputs = model(images)
            
            # Handle different model output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate probabilities and predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Process each sample
            for i in range(len(labels)):
                sample_idx += 1
                pred = preds[i].item()
                label = labels[i].item()
                prob = probs[i].cpu().numpy()
                img_path = paths[i]
                
                is_correct = (pred == label)
                if is_correct:
                    correct_count += 1
                
                # Get Top-5 predictions
                top5_indices = np.argsort(prob)[-5:][::-1]
                top5_probs = prob[top5_indices]
                
                # Save prediction details
                pred_detail = {
                    'sample_idx': sample_idx,
                    'image_path': img_path,
                    'true_label': label,
                    'true_class': idx_to_class.get(label, f"class_{label}"),
                    'pred_label': pred,
                    'pred_class': idx_to_class.get(pred, f"class_{pred}"),
                    'is_correct': is_correct,
                    'confidence': float(prob[pred]),
                    'top5': [
                        {
                            'rank': rank + 1,
                            'class_idx': int(idx),
                            'class_path': idx_to_class.get(idx, f"class_{idx}"),
                            'class_name': get_class_short_name(idx_to_class.get(idx, f"class_{idx}")),
                            'rock_type': get_rock_type_en(idx_to_class.get(idx, f"class_{idx}")),
                            'confidence': float(top5_probs[rank])
                        }
                        for rank, idx in enumerate(top5_indices)
                    ]
                }
                predictions_detail.append(pred_detail)
                
                status = "[OK]" if is_correct else "[X]"
                current_acc = correct_count / sample_idx * 100
                print(f"\n[{sample_idx}/{total_samples}] {status}")
                print(f"Image: {img_path}")
                print("-" * 60)
                print(f"{'Rank':<6}{'Class Name':<25}{'Rock Type':<15}{'Confidence':<12}")
                print("-" * 60)
                for rank, idx in enumerate(top5_indices):
                    class_path = idx_to_class.get(idx, f"class_{idx}")
                    class_name = get_class_short_name(class_path)
                    rock_type = get_rock_type_en(class_path)
                    conf = top5_probs[rank] * 100
                    marker = " <-- True" if idx == label else ""
                    print(f"{rank+1:<6}{class_name:<25}{rock_type:<15}{conf:.2f}%{marker}")
                print("-" * 60)
                print(f"Current Accuracy: {correct_count}/{sample_idx} = {current_acc:.2f}%")
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate various metrics
    results = compute_metrics(all_preds, all_labels, all_probs, class_mapping, predictions_detail)
    results['predictions_detail'] = predictions_detail
    
    return results


def compute_metrics(preds, labels, probs, class_mapping, predictions_detail=None):
    """
    Calculate evaluation metrics
    
    Args:
        preds: Prediction results
        labels: True labels
        probs: Prediction probabilities
        class_mapping: Class mapping
        predictions_detail: Detailed prediction information
    
    Returns:
        Dictionary containing various metrics
    """
    results = {}
    
    # 1. Overall accuracy
    correct = (preds == labels).sum()
    total = len(labels)
    results['overall_accuracy'] = correct / total
    
    # 2. Top-K accuracy
    for k in [1, 3, 5]:
        top_k_correct = 0
        for i, (label, prob) in enumerate(zip(labels, probs)):
            top_k_indices = np.argsort(prob)[-k:][::-1]
            if label in top_k_indices:
                top_k_correct += 1
        results[f'top_{k}_accuracy'] = top_k_correct / total
    
    # 3. Calculate TP, FP, FN for each class
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Initialize statistics
    class_tp = defaultdict(int)  # True Positives
    class_fp = defaultdict(int)  # False Positives
    class_fn = defaultdict(int)  # False Negatives
    class_total = defaultdict(int)  # Total samples
    
    for pred, label in zip(preds, labels):
        class_total[label] += 1
        if pred == label:
            class_tp[label] += 1
        else:
            class_fn[label] += 1  # False negative for label class
            class_fp[pred] += 1   # False positive for predicted class
    
    # 4. Calculate precision, recall, F1 for each class
    class_metrics = {}
    all_precisions = []
    all_recalls = []
    all_f1s = []
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    total_weight = sum(class_total.values())
    
    for class_idx in sorted(class_mapping.values()):
        class_path = idx_to_class.get(class_idx, f"class_{class_idx}")
        class_name = get_class_short_name(class_path)
        
        tp = class_tp[class_idx]
        fp = class_fp[class_idx]
        fn = class_fn[class_idx]
        n_samples = class_total[class_idx]
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Recall: TP / (TP + FN) = TP / total_of_class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_path] = {
            'class_name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total': n_samples,
            'correct': tp
        }
        
        if n_samples > 0:
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            
            # Weighted statistics
            weighted_precision += precision * n_samples
            weighted_recall += recall * n_samples
            weighted_f1 += f1 * n_samples
    
    results['class_metrics'] = class_metrics
    
    # 5. Macro Average
    results['macro_precision'] = np.mean(all_precisions) if all_precisions else 0
    results['macro_recall'] = np.mean(all_recalls) if all_recalls else 0
    results['macro_f1'] = np.mean(all_f1s) if all_f1s else 0
    
    # 6. Weighted Average
    results['weighted_precision'] = weighted_precision / total_weight if total_weight > 0 else 0
    results['weighted_recall'] = weighted_recall / total_weight if total_weight > 0 else 0
    results['weighted_f1'] = weighted_f1 / total_weight if total_weight > 0 else 0
    
    # 7. Per-class accuracy (compatible with old format)
    class_accuracies = {}
    for class_path, metrics in class_metrics.items():
        class_accuracies[class_path] = {
            'accuracy': metrics['recall'],  # Recall is class accuracy
            'correct': metrics['correct'],
            'total': metrics['total']
        }
    results['class_accuracies'] = class_accuracies
    
    # 8. Major category accuracy
    category_results = {}
    for category, indices in ROCK_CATEGORIES.items():
        cat_correct = sum(class_tp[idx] for idx in indices)
        cat_total = sum(class_total[idx] for idx in indices)
        acc = cat_correct / cat_total if cat_total > 0 else 0
        category_results[category] = {
            'accuracy': acc,
            'correct': cat_correct,
            'total': cat_total,
            'name': CATEGORY_NAMES[category]
        }
    results['category_accuracies'] = category_results
    
    # 9. Confusion matrix
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for pred, label in zip(preds, labels):
        confusion_matrix[label, pred] += 1
    results['confusion_matrix'] = confusion_matrix.tolist()
    
    # 10. Statistics
    results['total_samples'] = total
    results['correct_samples'] = int(correct)
    
    # 11. Wrong predictions summary
    if predictions_detail:
        wrong_predictions = [p for p in predictions_detail if not p['is_correct']]
        results['wrong_predictions'] = wrong_predictions
        results['wrong_count'] = len(wrong_predictions)
    
    # 12. Worst performing classes (sorted by F1)
    worst_classes = sorted(
        [(k, v) for k, v in class_metrics.items() if v['total'] > 0],
        key=lambda x: x[1]['f1']
    )[:5]
    results['worst_classes'] = [
        {
            'class_path': k,
            'class_name': v['class_name'],
            'f1': v['f1'],
            'recall': v['recall'],
            'precision': v['precision'],
            'total': v['total']
        }
        for k, v in worst_classes
    ]
    
    return results


# ============================================================================
# Report Generation
# ============================================================================

def print_results(model_name, results, checkpoint_path=None):
    """Print formatted test results"""
    print("\n" + "=" * 70)
    print("Model Evaluation Report")
    print("=" * 70)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_name}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    
    # Overall metrics
    print("\n" + "-" * 70)
    print("Overall Metrics")
    print("-" * 70)
    print(f"Test samples: {results['total_samples']}")
    print(f"Correct predictions: {results['correct_samples']}")
    print(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
    print()
    print(f"Top-1 Accuracy: {results['top_1_accuracy']*100:.2f}%")
    print(f"Top-3 Accuracy: {results['top_3_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy: {results['top_5_accuracy']*100:.2f}%")
    print()
    print(f"Macro Precision: {results['macro_precision']*100:.2f}%")
    print(f"Macro Recall: {results['macro_recall']*100:.2f}%")
    print(f"Macro F1 Score: {results['macro_f1']*100:.2f}%")
    print()
    print(f"Weighted Precision: {results['weighted_precision']*100:.2f}%")
    print(f"Weighted Recall: {results['weighted_recall']*100:.2f}%")
    print(f"Weighted F1 Score: {results['weighted_f1']*100:.2f}%")
    
    # Major category accuracy
    print("\n" + "-" * 70)
    print("Statistics by Rock Major Category")
    print("-" * 70)
    for category, stats in results['category_accuracies'].items():
        print(f"{stats['name']} ({category.replace('Rocks', '')}): "
              f"{stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Per-class detailed metrics
    print("\n" + "-" * 70)
    print("Per-Class Detailed Metrics")
    print("-" * 70)
    print(f"{'Class Name':<30}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'Samples':<8}")
    print("-" * 70)
    
    # Sort by F1 score in descending order
    sorted_metrics = sorted(
        results['class_metrics'].items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    
    for class_path, metrics in sorted_metrics:
        class_name = metrics['class_name']
        print(f"{class_name:<30}"
              f"{metrics['precision']*100:>8.2f}%   "
              f"{metrics['recall']*100:>8.2f}%   "
              f"{metrics['f1']*100:>8.2f}%   "
              f"{metrics['total']:>5}")
    print("-" * 70)
    
    # Worst performing classes
    if 'worst_classes' in results and results['worst_classes']:
        print("\n" + "-" * 70)
        print("Top 5 Worst Performing Classes (Need Improvement)")
        print("-" * 70)
        for wc in results['worst_classes']:
            print(f"  - {wc['class_name']}: F1={wc['f1']*100:.2f}%, "
                  f"Recall={wc['recall']*100:.2f}%, Precision={wc['precision']*100:.2f}%")
    
    # Wrong predictions summary
    if 'wrong_predictions' in results and results['wrong_predictions']:
        print("\n" + "=" * 70)
        print(f"Wrong Predictions Summary ({len(results['wrong_predictions'])} items)")
        print("=" * 70)
        for wp in results['wrong_predictions']:
            img_name = os.path.basename(wp['image_path'])
            true_class = get_class_short_name(wp['true_class'])
            pred_class = get_class_short_name(wp['pred_class'])
            conf = wp['confidence'] * 100
            print(f"\n  {img_name}:")
            print(f"    True: {true_class}")
            print(f"    Predicted: {pred_class} ({conf:.2f}%)")
    
    print("\n" + "=" * 70)


def save_report(results_dict, output_dir, timestamp):
    """
    Save test report
    
    Args:
        results_dict: Test results for all models
        output_dir: Output directory
        timestamp: Timestamp
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, results in results_dict.items():
        # Prepare serializable results
        serializable_results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'overall_accuracy': float(results['overall_accuracy']),
            'top_1_accuracy': float(results['top_1_accuracy']),
            'top_3_accuracy': float(results['top_3_accuracy']),
            'top_5_accuracy': float(results['top_5_accuracy']),
            'macro_precision': float(results['macro_precision']),
            'macro_recall': float(results['macro_recall']),
            'macro_f1': float(results['macro_f1']),
            'weighted_precision': float(results['weighted_precision']),
            'weighted_recall': float(results['weighted_recall']),
            'weighted_f1': float(results['weighted_f1']),
            'total_samples': results['total_samples'],
            'correct_samples': results['correct_samples'],
            'wrong_count': results.get('wrong_count', 0),
            'category_accuracies': {
                k: {
                    'accuracy': float(v['accuracy']),
                    'correct': v['correct'],
                    'total': v['total'],
                    'name': v['name']
                }
                for k, v in results['category_accuracies'].items()
            },
            'class_metrics': {
                k: {
                    'class_name': v['class_name'],
                    'precision': float(v['precision']),
                    'recall': float(v['recall']),
                    'f1': float(v['f1']),
                    'total': v['total'],
                    'correct': v['correct']
                }
                for k, v in results['class_metrics'].items()
            },
            'worst_classes': results.get('worst_classes', []),
            'confusion_matrix': results['confusion_matrix']
        }
        
        # Add prediction details (if exists)
        if 'predictions_detail' in results:
            serializable_results['predictions_detail'] = results['predictions_detail']
        
        # Save JSON report
        json_path = os.path.join(output_dir, f'test_report_{model_name}_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report saved: {json_path}")
        
        # Save detailed text report
        txt_path = os.path.join(output_dir, f'test_result_{model_name}_{timestamp}.txt')
        save_detailed_text_report(results, model_name, txt_path, timestamp)
        print(f"Detailed report saved: {txt_path}")


def save_detailed_text_report(results, model_name, output_path, timestamp):
    """Save detailed text format report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Model Evaluation Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write(f"Model: {model_name}\n\n")
        
        # Overall metrics
        f.write("-" * 70 + "\n")
        f.write("Overall Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(f"Test samples: {results['total_samples']}\n")
        f.write(f"Correct predictions: {results['correct_samples']}\n")
        f.write(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}%\n\n")
        f.write(f"Top-1 Accuracy: {results['top_1_accuracy']*100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {results['top_3_accuracy']*100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {results['top_5_accuracy']*100:.2f}%\n\n")
        f.write(f"Macro Precision: {results['macro_precision']*100:.2f}%\n")
        f.write(f"Macro Recall: {results['macro_recall']*100:.2f}%\n")
        f.write(f"Macro F1 Score: {results['macro_f1']*100:.2f}%\n\n")
        f.write(f"Weighted Precision: {results['weighted_precision']*100:.2f}%\n")
        f.write(f"Weighted Recall: {results['weighted_recall']*100:.2f}%\n")
        f.write(f"Weighted F1 Score: {results['weighted_f1']*100:.2f}%\n\n")
        
        # Major category accuracy
        f.write("-" * 70 + "\n")
        f.write("Statistics by Rock Major Category\n")
        f.write("-" * 70 + "\n")
        for category, stats in results['category_accuracies'].items():
            f.write(f"{stats['name']} ({category.replace('Rocks', '')}): "
                   f"{stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})\n")
        f.write("\n")
        
        # Per-class detailed metrics
        f.write("-" * 70 + "\n")
        f.write("Per-Class Detailed Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class Name':<30}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'Samples':<8}\n")
        f.write("-" * 70 + "\n")
        
        sorted_metrics = sorted(
            results['class_metrics'].items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )
        
        for class_path, metrics in sorted_metrics:
            class_name = metrics['class_name']
            f.write(f"{class_name:<30}"
                   f"{metrics['precision']*100:>8.2f}%   "
                   f"{metrics['recall']*100:>8.2f}%   "
                   f"{metrics['f1']*100:>8.2f}%   "
                   f"{metrics['total']:>5}\n")
        f.write("-" * 70 + "\n\n")
        
        # Worst performing classes
        if 'worst_classes' in results and results['worst_classes']:
            f.write("-" * 70 + "\n")
            f.write("Top 5 Worst Performing Classes (Need Improvement)\n")
            f.write("-" * 70 + "\n")
            for wc in results['worst_classes']:
                f.write(f"  - {wc['class_name']}: F1={wc['f1']*100:.2f}%, "
                       f"Recall={wc['recall']*100:.2f}%\n")
            f.write("\n")
        
        # Detailed prediction results
        if 'predictions_detail' in results:
            f.write("=" * 70 + "\n")
            f.write("Detailed Prediction Results\n")
            f.write("=" * 70 + "\n\n")
            
            for pred in results['predictions_detail']:
                status = "[OK]" if pred['is_correct'] else "[X]"
                img_name = os.path.basename(pred['image_path'])
                f.write(f"[{pred['sample_idx']}] {status} {img_name}\n")
                f.write(f"    Path: {pred['image_path']}\n")
                f.write(f"    True: {get_class_short_name(pred['true_class'])} (ID: {pred['true_label']})\n")
                f.write(f"    Predicted: {get_class_short_name(pred['pred_class'])} (Confidence: {pred['confidence']*100:.2f}%)\n")
                
                # Top-3 predictions
                top3_str = ", ".join([
                    f"{t['class_name']}({t['confidence']*100:.1f}%)"
                    for t in pred['top5'][:3]
                ])
                f.write(f"    Top-3: {top3_str}\n\n")
        
        # Wrong predictions summary
        if 'wrong_predictions' in results and results['wrong_predictions']:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Wrong Predictions Summary ({len(results['wrong_predictions'])} items)\n")
            f.write("=" * 70 + "\n\n")
            
            for wp in results['wrong_predictions']:
                img_name = os.path.basename(wp['image_path'])
                true_class = get_class_short_name(wp['true_class'])
                pred_class = get_class_short_name(wp['pred_class'])
                conf = wp['confidence'] * 100
                f.write(f"  {img_name}:\n")
                f.write(f"    True: {true_class}\n")
                f.write(f"    Predicted: {pred_class} ({conf:.2f}%)\n\n")


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Rock Classification Model Testing Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python src/test_model.py --model ensemble
  python src/test_model.py --model resnet_optimized
  python src/test_model.py --model all
  python src/test_model.py --model all --output_dir my_reports
        """
    )
    
    parser.add_argument(
        '--model', type=str, default='all',
        choices=['ensemble', 'resnet_optimized', 'all'],
        help='Model type to test (default: all)'
    )
    
    parser.add_argument(
        '--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
        help=f'Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})'
    )
    
    parser.add_argument(
        '--test_data_dir', type=str, default=DEFAULT_TEST_DATA_DIR,
        help=f'Test data directory (default: {DEFAULT_TEST_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Report output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size (default: 16)'
    )
    
    parser.add_argument(
        '--no_save', action='store_true',
        help='Do not save report files'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 80)
    print("Rock Classification Model Testing Script")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Discover available models
    available_models = discover_models(args.checkpoint_dir)
    print(f"\nDiscovered models: {list(available_models.keys())}")
    
    if not available_models:
        print(f"Error: No model files found in {args.checkpoint_dir} directory")
        print("Please ensure ensemble.pth or resnet50_optimized.pth files exist")
        sys.exit(1)
    
    # Determine models to test
    if args.model == 'all':
        models_to_test = available_models
    else:
        if args.model not in available_models:
            print(f"Error: Checkpoint file for {args.model} model not found")
            print(f"Available models: {list(available_models.keys())}")
            sys.exit(1)
        models_to_test = {args.model: available_models[args.model]}
    
    # Load test data
    print(f"\nLoading test data: {args.test_data_dir}")
    test_dataset = TestDataset(
        data_dir=args.test_data_dir,
        transform=get_test_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Test each model
    results_dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, checkpoint_path in models_to_test.items():
        print(f"\n{'=' * 40}")
        print(f"Starting test: {model_name}")
        print(f"{'=' * 40}")
        
        try:
            # Load model
            model = load_model(model_name, checkpoint_path, device)
            
            # Evaluate model
            print("Evaluating model...")
            results = evaluate_model(
                model, test_loader, device, 
                test_dataset.class_mapping
            )
            results_dict[model_name] = results
            
            # Print results
            print_results(model_name, results, checkpoint_path)
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save report
    if results_dict and not args.no_save:
        save_report(results_dict, args.output_dir, timestamp)
    
    # Print comparison summary (if multiple models tested)
    if len(results_dict) > 1:
        print("\n" + "=" * 80)
        print("Model Performance Comparison")
        print("=" * 80)
        print(f"{'Model':<20} {'Top-1 Accuracy':<15} {'Top-3 Accuracy':<15} {'Top-5 Accuracy':<15}")
        print("-" * 65)
        for model_name, results in results_dict.items():
            print(f"{model_name:<20} "
                  f"{results['top_1_accuracy']*100:.2f}%{'':<10} "
                  f"{results['top_3_accuracy']*100:.2f}%{'':<10} "
                  f"{results['top_5_accuracy']*100:.2f}%")
        print("=" * 80)
    
    print("\nTesting completed!")


if __name__ == "__main__":
    main()

