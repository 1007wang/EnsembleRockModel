"""
Rock Classification Model Testing Script

Features:
- Load trained model checkpoints
- Predict on single or batch images
- Output Top-K prediction results
- Support JSON format output
- Calculate and output model evaluation metrics (accuracy, precision, recall, F1-score)
- Automatically save results to file

Usage Examples:
    # Single image prediction
    python test_model.py --image test_data/IgneousRocks/andesite/IMG_20201119_173036.jpg

    # Batch prediction (automatically calculate metrics and save results)
    python test_model.py --image_dir test_data/

    # JSON output
    python test_model.py --image test_data/SedimentaryRocks/limestone/IMG_20201119_165623.jpg --json

    # Quiet mode
    python test_model.py --image_dir test_data/ --quiet

    # Specify checkpoint
    python test_model.py --image_dir test_data/ --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from datetime import datetime
from collections import defaultdict

# Resolve OpenMP library conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ensemble_model import EnsembleModel


def get_class_basename(class_name):
    """Extract base name from class path"""
    if '\\' in class_name:
        return class_name.split('\\')[-1]
    elif '/' in class_name:
        return class_name.split('/')[-1]
    return class_name


def get_rock_type(class_name):
    """Extract major rock type from class path"""
    if class_name.startswith('IgneousRocks'):
        return 'Igneous'
    elif class_name.startswith('MetamorphicRocks'):
        return 'Metamorphic'
    elif class_name.startswith('SedimentaryRocks'):
        return 'Sedimentary'
    return 'Unknown'


def extract_label_from_path(image_path, class_mapping):
    """
    Extract true label from image path
    
    Assumes path format: .../rock_type/subclass/image.jpg
    Example: test_data/IgneousRocks/andesite/IMG_001.jpg
    
    Args:
        image_path: Image path
        class_mapping: Class mapping {class_path: class_id}
    
    Returns:
        (class_id, class_path) or (None, None) if extraction fails
    """
    # Normalize path separators
    normalized_path = image_path.replace('\\', '/')
    parts = normalized_path.split('/')
    
    # Rock type directory names
    rock_type_dirs = ('IgneousRocks', 'MetamorphicRocks', 'SedimentaryRocks')
    
    # Try to match rock_type/subclass structure in path
    for i in range(len(parts) - 2):
        rock_dir = parts[i]
        subclass = parts[i+1]
        
        # Check if it's a rock type directory
        if rock_dir in rock_type_dirs:
            # Try both backslash and forward slash formats
            potential_class_path = f"{rock_dir}\\{subclass}"
            potential_class_path_alt = f"{rock_dir}/{subclass}"
            
            if potential_class_path in class_mapping:
                return class_mapping[potential_class_path], potential_class_path
            elif potential_class_path_alt in class_mapping:
                return class_mapping[potential_class_path_alt], potential_class_path_alt
    
    return None, None


def calculate_metrics(results, class_mapping):
    """
    Calculate classification metrics
    
    Args:
        results: List of prediction results, each containing predictions and true_label
        class_mapping: Class mapping
    
    Returns:
        metrics: Metrics dictionary
    """
    # Collect true labels and predicted labels
    y_true = []
    y_pred = []
    y_pred_top3 = []  # Top-3 predictions
    y_pred_top5 = []  # Top-5 predictions
    
    valid_count = 0
    
    for result in results:
        true_label = result.get('true_label')
        if true_label is None:
            continue
        
        valid_count += 1
        pred_label = result['predictions'][0]['class_id']
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        # Top-3 and Top-5 predictions
        top3_ids = [p['class_id'] for p in result['predictions'][:3]]
        top5_ids = [p['class_id'] for p in result['predictions'][:5]]
        y_pred_top3.append(top3_ids)
        y_pred_top5.append(top5_ids)
    
    if valid_count == 0:
        return None
    
    # Basic accuracy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = np.mean(y_true == y_pred)
    
    # Top-3 and Top-5 accuracy
    top3_correct = sum(1 for true, preds in zip(y_true, y_pred_top3) if true in preds)
    top5_correct = sum(1 for true, preds in zip(y_true, y_pred_top5) if true in preds)
    top3_accuracy = top3_correct / len(y_true)
    top5_accuracy = top5_correct / len(y_true)
    
    # Metrics per class
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Calculate precision, recall, F1 for each class
    class_metrics = {}
    id_to_name = {v: k for k, v in class_mapping.items()}
    
    for cls in unique_classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = get_class_basename(id_to_name.get(cls, f'Unknown_{cls}'))
        class_metrics[cls] = {
            'name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(y_true == cls))
        }
    
    # Macro average metrics
    precisions = [m['precision'] for m in class_metrics.values() if m['support'] > 0]
    recalls = [m['recall'] for m in class_metrics.values() if m['support'] > 0]
    f1s = [m['f1'] for m in class_metrics.values() if m['support'] > 0]
    
    macro_precision = np.mean(precisions) if precisions else 0
    macro_recall = np.mean(recalls) if recalls else 0
    macro_f1 = np.mean(f1s) if f1s else 0
    
    # Weighted average metrics
    total_support = sum(m['support'] for m in class_metrics.values())
    weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    
    # Statistics by rock type
    rock_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for result in results:
        true_label = result.get('true_label')
        if true_label is None:
            continue
        
        true_path = result.get('true_class_path', '')
        pred_label = result['predictions'][0]['class_id']
        
        rock_type = get_rock_type(true_path)
        rock_type_stats[rock_type]['total'] += 1
        if true_label == pred_label:
            rock_type_stats[rock_type]['correct'] += 1
    
    rock_type_accuracy = {}
    for rock_type, stats in rock_type_stats.items():
        if stats['total'] > 0:
            rock_type_accuracy[rock_type] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    return {
        'total_samples': len(y_true),
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics,
        'rock_type_accuracy': rock_type_accuracy
    }


def format_metrics_report(metrics, checkpoint_path=None):
    """
    Format metrics report
    
    Args:
        metrics: Metrics dictionary
        checkpoint_path: Checkpoint path
    
    Returns:
        formatted_report: Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Model Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if checkpoint_path:
        lines.append(f"Model checkpoint: {checkpoint_path}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("Overall Metrics")
    lines.append("-" * 70)
    lines.append(f"Test samples: {metrics['total_samples']}")
    lines.append(f"Top-1 Accuracy: {metrics['accuracy']*100:.2f}%")
    lines.append(f"Top-3 Accuracy: {metrics['top3_accuracy']*100:.2f}%")
    lines.append(f"Top-5 Accuracy: {metrics['top5_accuracy']*100:.2f}%")
    lines.append("")
    lines.append(f"Macro Precision: {metrics['macro_precision']*100:.2f}%")
    lines.append(f"Macro Recall: {metrics['macro_recall']*100:.2f}%")
    lines.append(f"Macro F1: {metrics['macro_f1']*100:.2f}%")
    lines.append("")
    lines.append(f"Weighted Precision: {metrics['weighted_precision']*100:.2f}%")
    lines.append(f"Weighted Recall: {metrics['weighted_recall']*100:.2f}%")
    lines.append(f"Weighted F1: {metrics['weighted_f1']*100:.2f}%")
    lines.append("")
    
    # Statistics by rock type
    if metrics.get('rock_type_accuracy'):
        lines.append("-" * 70)
        lines.append("Statistics by Major Rock Type")
        lines.append("-" * 70)
        for rock_type, stats in sorted(metrics['rock_type_accuracy'].items()):
            lines.append(f"{rock_type}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
        lines.append("")
    
    # Detailed metrics per class
    lines.append("-" * 70)
    lines.append("Detailed Metrics by Class")
    lines.append("-" * 70)
    lines.append(f"{'Class Name':<30}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'Support':<10}")
    lines.append("-" * 70)
    
    # Sort by F1 score for display
    sorted_classes = sorted(
        metrics['class_metrics'].items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    
    for cls_id, cls_metrics in sorted_classes:
        name = cls_metrics['name'][:28]
        lines.append(
            f"{name:<30}"
            f"{cls_metrics['precision']*100:>8.2f}%   "
            f"{cls_metrics['recall']*100:>8.2f}%   "
            f"{cls_metrics['f1']*100:>8.2f}%   "
            f"{cls_metrics['support']:>6}"
        )
    
    lines.append("-" * 70)
    lines.append("")
    
    # Find worst performing classes
    worst_classes = sorted(
        [(cls_id, m) for cls_id, m in metrics['class_metrics'].items() if m['support'] > 0],
        key=lambda x: x[1]['f1']
    )[:5]
    
    if worst_classes:
        lines.append("-" * 70)
        lines.append("Worst Performing 5 Classes (Need Improvement)")
        lines.append("-" * 70)
        for cls_id, cls_metrics in worst_classes:
            lines.append(f"  - {cls_metrics['name']}: F1={cls_metrics['f1']*100:.2f}%, Recall={cls_metrics['recall']*100:.2f}%")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_results_to_file(results, metrics, output_file, checkpoint_path=None):
    """
    Save results to file
    
    Args:
        results: List of prediction results
        metrics: Metrics dictionary
        output_file: Output file path
        checkpoint_path: Checkpoint path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write metrics report
        if metrics:
            f.write(format_metrics_report(metrics, checkpoint_path))
            f.write("\n\n")
        
        # Write detailed prediction results
        f.write("=" * 70 + "\n")
        f.write("Detailed Prediction Results\n")
        f.write("=" * 70 + "\n\n")
        
        # Calculate and display accuracy summary at the beginning
        if metrics:
            f.write("-" * 70 + "\n")
            f.write("Accuracy Summary\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Test Samples: {metrics['total_samples']}\n")
            f.write(f"Top-1 Accuracy: {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Top-3 Accuracy: {metrics['top3_accuracy']*100:.2f}%\n")
            f.write(f"Top-5 Accuracy: {metrics['top5_accuracy']*100:.2f}%\n")
            f.write("-" * 70 + "\n\n")
        
        correct_count = 0
        wrong_results = []
        
        for i, result in enumerate(results):
            image_path = result['image_path']
            predictions = result['predictions']
            true_label = result.get('true_label')
            true_class_path = result.get('true_class_path', 'Unknown')
            
            pred_label = predictions[0]['class_id']
            pred_class = predictions[0]['class_name']
            pred_conf = predictions[0]['confidence']
            
            is_correct = true_label == pred_label if true_label is not None else None
            
            if is_correct:
                correct_count += 1
                status = "✓"
            elif is_correct is False:
                status = "✗"
                wrong_results.append(result)
            else:
                status = "?"
            
            f.write(f"[{i+1}] {status} {os.path.basename(image_path)}\n")
            f.write(f"    Path: {image_path}\n")
            if true_label is not None:
                f.write(f"    True: {get_class_basename(true_class_path)} (ID: {true_label})\n")
            f.write(f"    Predicted: {pred_class} (Confidence: {pred_conf:.2f}%)\n")
            
            # Show Top-3 predictions
            if len(predictions) > 1:
                f.write(f"    Top-3: ")
                top3_str = ", ".join([f"{p['class_name']}({p['confidence']:.1f}%)" for p in predictions[:3]])
                f.write(top3_str + "\n")
            f.write("\n")
        
        # Write error prediction summary
        if wrong_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Error Prediction Summary ({len(wrong_results)} items)\n")
            f.write("=" * 70 + "\n\n")
            
            for result in wrong_results:
                image_path = result['image_path']
                true_class_path = result.get('true_class_path', 'Unknown')
                pred_class = result['predictions'][0]['class_name']
                pred_conf = result['predictions'][0]['confidence']
                
                f.write(f"  {os.path.basename(image_path)}:\n")
                f.write(f"    True: {get_class_basename(true_class_path)}\n")
                f.write(f"    Predicted: {pred_class} ({pred_conf:.2f}%)\n\n")
    
    print(f"\nResults saved to: {output_file}")


def load_model(checkpoint_path, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Model checkpoint path
        device: Device (cuda or cpu)
    
    Returns:
        model: Loaded model
        class_mapping: Class mapping
        checkpoint_info: Checkpoint information
    """
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load class mapping (from test_data directory)
    class_mapping_path = os.path.join('test_data', 'class_mapping.json')
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")
    
    with open(class_mapping_path, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    
    num_classes = len(class_mapping)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Define key prefixes that can be ignored during inference (training-specific modules)
    ignorable_prefixes = ['criterion.', 'loss.', 'optimizer.']
    
    # Filter out training-specific keys from checkpoint
    for key in list(state_dict.keys()):
        if any(key.startswith(prefix) for prefix in ignorable_prefixes):
            del state_dict[key]
    
    # Create EnsembleModel
    model = EnsembleModel(num_classes=num_classes)
    
    # Load weights (use non-strict mode to handle architecture differences)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Core backbone keys that must be loaded
    core_prefixes = ['inception.inception.', 'efficientnet.efficientnet.', 'inception.fc.', 'efficientnet.fc.']
    
    # Check if core weights are loaded
    critical_missing = [k for k in missing_keys if any(k.startswith(p) for p in core_prefixes)]
    
    if critical_missing:
        print(f"⚠ Warning: Missing critical model weights!")
        print(f"  Missing keys: {critical_missing[:5]}{'...' if len(critical_missing) > 5 else ''}")
    else:
        print("✓ Model weights loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    # Get checkpoint info
    checkpoint_info = {}
    if 'epoch' in checkpoint:
        checkpoint_info['epoch'] = checkpoint['epoch']
    
    return model, class_mapping, checkpoint_info


def get_transforms():
    """Get image preprocessing transforms"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])


def load_image(image_path):
    """
    Load and preprocess image
    
    Args:
        image_path: Image path
    
    Returns:
        img_tensor: Preprocessed image tensor
        img_original: Original PIL image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path)
    
    # Ensure RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save original image for heatmap
    img_original = img.copy()
    
    # Apply transforms
    transform = get_transforms()
    img_tensor = transform(img)
    
    return img_tensor, img_original


def predict_single(model, img_tensor, class_mapping, device, top_k=5):
    """
    Predict on a single image
    
    Args:
        model: Model
        img_tensor: Image tensor
        class_mapping: Class mapping
        device: Device
        top_k: Return top-k prediction results
    
    Returns:
        predictions: List of prediction results
    """
    # Create reverse mapping
    id_to_name = {v: k for k, v in class_mapping.items()}
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # Handle different model output formats
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        # Calculate probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)
        
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            class_path = id_to_name[idx.item()]
            class_name = get_class_basename(class_path)
            rock_type = get_rock_type(class_path)
            
            predictions.append({
                'rank': i + 1,
                'class_id': idx.item(),
                'class_name': class_name,
                'class_path': class_path,
                'rock_type': rock_type,
                'confidence': prob.item() * 100
            })
    
    return predictions


def print_predictions(predictions, image_path=None, json_output=False):
    """Print prediction results"""
    if json_output:
        result = {
            'image_path': image_path,
            'predictions': predictions
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if image_path:
            print(f"\nImage: {image_path}")
        print("-" * 60)
        print(f"{'Rank':<6}{'Class Name':<25}{'Rock Type':<20}{'Confidence':<10}")
        print("-" * 60)
        for pred in predictions:
            print(f"{pred['rank']:<6}{pred['class_name']:<25}{pred['rock_type']:<20}{pred['confidence']:.2f}%")
        print("-" * 60)


def predict_batch(model, image_dir, class_mapping, device, top_k=5, 
                  json_output=False, extract_labels=True, verbose=True):
    """
    Batch predict on all images in directory
    
    Args:
        model: Model
        image_dir: Image directory
        class_mapping: Class mapping
        device: Device
        top_k: Return top-k prediction results
        json_output: Whether to output in JSON format
        extract_labels: Whether to extract true labels from path
        verbose: Whether to print detailed information
    
    Returns:
        all_results: All prediction results
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Collect all images
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No valid image files found in directory: {image_dir}")
        return []
    
    all_results = []
    correct_count = 0
    labeled_count = 0
    
    for i, image_path in enumerate(image_files):
        try:
            img_tensor, _ = load_image(image_path)
            predictions = predict_single(model, img_tensor, class_mapping, device, top_k)
            
            result = {
                'image_path': image_path,
                'predictions': predictions
            }
            
            # Try to extract true label from path
            if extract_labels:
                true_label, true_class_path = extract_label_from_path(image_path, class_mapping)
                if true_label is not None:
                    result['true_label'] = true_label
                    result['true_class_path'] = true_class_path
                    labeled_count += 1
                    
                    # Check if prediction is correct
                    if predictions[0]['class_id'] == true_label:
                        correct_count += 1
                        result['correct'] = True
                    else:
                        result['correct'] = False
            
            all_results.append(result)
            
            if verbose and not json_output:
                # Show progress and prediction results
                status = ""
                if 'correct' in result:
                    status = " ✓" if result['correct'] else " ✗"
                print(f"\n[{i+1}/{len(image_files)}]{status} ", end='')
                print_predictions(predictions, image_path, json_output=False)
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue
    
    if json_output:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
    elif labeled_count > 0:
        # Print brief statistics
        print(f"\n{'='*60}")
        print(f"Prediction completed: {correct_count}/{labeled_count} correct (Accuracy: {correct_count/labeled_count*100:.2f}%)")
        print(f"{'='*60}")
    
    return all_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Rock Classification Model Testing Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Single image prediction (using default checkpoint)
  python test_model.py --image test_data/IgneousRocks/andesite/IMG_20201119_173036.jpg

  # Batch prediction (automatically calculate metrics and save results)
  python test_model.py --image_dir test_data/

  # JSON format output
  python test_model.py --image test_data/SedimentaryRocks/limestone/IMG_20201119_165623.jpg --json

  # Quiet mode
  python test_model.py --image_dir test_data/ --quiet

  # Specify checkpoint
  python test_model.py --image_dir test_data/ --checkpoint checkpoints/best_model.pth
        """
    )
    
    # Input parameters
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Single image path')
    input_group.add_argument('--image_dir', type=str, help='Image directory path (batch prediction)')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Model checkpoint path (.pth file, default: checkpoints/best_model.pth)')
    
    # Output parameters
    parser.add_argument('--top_k', type=int, default=5,
                        help='Output Top-K prediction results (default: 5)')
    parser.add_argument('--json', action='store_true',
                        help='Output results in JSON format')
    
    # Output and metrics parameters
    parser.add_argument('--output_file', '-o', type=str, default='auto',
                        help='Output results to file (default: auto-generate timestamped filename, set to "none" to disable)')
    parser.add_argument('--no_metrics', action='store_true',
                        help='Disable evaluation metrics calculation (metrics calculated by default)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode, reduce output')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Computing device (cuda/cpu, default: auto-detect)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle output file path
    if args.output_file == 'auto':
        # Auto-generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('test_reports', exist_ok=True)
        args.output_file = f'test_reports/test_result_{timestamp}.txt'
    elif args.output_file and args.output_file.lower() == 'none':
        args.output_file = None
    
    # Whether to calculate metrics (calculate by default)
    compute_metrics = not args.no_metrics
    
    if not args.json:
        print(f"\n{'='*60}")
        print("Rock Classification Model Testing")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Checkpoint: {args.checkpoint}")
        if args.output_file:
            print(f"Output file: {args.output_file}")
        print(f"{'='*60}")
    
    try:
        # Load model
        model, class_mapping, checkpoint_info = load_model(
            args.checkpoint, 
            device
        )
        
        if not args.json:
            print(f"Number of classes: {len(class_mapping)}")
            print("✓ Model loaded successfully!")
            print(f"{'='*60}")
        
        # Single image prediction
        if args.image:
            img_tensor, _ = load_image(args.image)
            predictions = predict_single(model, img_tensor, class_mapping, device, args.top_k)
            
            # Try to extract true label
            true_label, true_class_path = extract_label_from_path(args.image, class_mapping)
            
            result = {
                'image_path': args.image,
                'predictions': predictions
            }
            if true_label is not None:
                result['true_label'] = true_label
                result['true_class_path'] = true_class_path
            
            print_predictions(predictions, args.image, args.json)
            
            # Show true label comparison
            if true_label is not None and not args.json:
                is_correct = predictions[0]['class_id'] == true_label
                status = "✓ Correct" if is_correct else "✗ Incorrect"
                print(f"True class: {get_class_basename(true_class_path)}")
                print(f"Prediction result: {status}")
            
            # Save to file (single image not saved by default unless explicitly specified)
            if args.output_file and args.output_file != f'test_reports/test_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt':
                save_results_to_file([result], None, args.output_file, args.checkpoint)
        
        # Batch prediction
        elif args.image_dir:
            verbose = not args.quiet and not args.json
            
            results = predict_batch(
                model, 
                args.image_dir, 
                class_mapping, 
                device, 
                args.top_k,
                args.json,
                extract_labels=True,
                verbose=verbose
            )
            
            # Calculate metrics (calculate by default)
            metrics = None
            if compute_metrics:
                metrics = calculate_metrics(results, class_mapping)
                
                if metrics and not args.json:
                    print("\n" + format_metrics_report(metrics, args.checkpoint))
            
            # Save to file (save by default)
            if args.output_file:
                save_results_to_file(results, metrics, args.output_file, args.checkpoint)
        
        if not args.json:
            print(f"\n{'='*60}")
            print("Testing completed!")
            print(f"{'='*60}\n")
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

