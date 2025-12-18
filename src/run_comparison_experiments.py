"""
Comparison experiment script - Automatically run comparison experiments for different model configurations
Compare in a systematic and progressive manner to clearly demonstrate the contribution of each component

Supported models:
- resnet50: Original ResNet50 model
- resnet50_optimized: Optimized ResNet50 model (multi-scale attention + hard example mining + triangular loss)
- efficientnet_b4: EfficientNet-B4 model
- inceptionv3: InceptionV3 model
- ensemble: Ensemble model

Experiment stages:
1. All models (no attention) + cosine scheduler
2. All models (no attention) + multi-stage scheduler  
3. All models (with attention) + cosine scheduler
4. All models (with attention) + multi-stage scheduler
5. ResNet50 optimized version comparison (only when resnet50_optimized is included)

Usage examples:
1. Run complete comparison experiments for all models:
   python run_comparison_experiments.py --data_dir /path/to/data

2. Test only optimized ResNet50:
   python run_comparison_experiments.py --optimized_only --data_dir /path/to/data

3. Compare only original ResNet50 with optimized version:
   python run_comparison_experiments.py --comparison_only --data_dir /path/to/data

4. Run experiments for specific stages:
   python run_comparison_experiments.py --start_stage 3 --end_stage 5 --data_dir /path/to/data

5. Skip certain stages:
   python run_comparison_experiments.py --skip_stages 1 2 --data_dir /path/to/data

6. Run optimized ResNet50 with custom parameters:
   python run_comparison_experiments.py --optimized_only --optimized_batch_size 12 --optimized_lr 5e-5 --data_dir /path/to/data
"""
import os
import subprocess
import argparse
import json
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Comparison experiment script - Systematically run comparison experiments for different model configurations')
    
    # Basic training parameters
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience value')
    
    # Experiment control parameters
    parser.add_argument('--basic_models', nargs='+', 
                       default=['resnet50', 'resnet50_optimized', 'efficientnet_b4', 'inceptionv3'],
                       help='List of base models to run')
    parser.add_argument('--skip_stages', nargs='+', default=[],
                       help='Experiment stages to skip, from 1 to 4')
    parser.add_argument('--start_stage', type=int, default=1,
                       help='Which stage to start experiments from (1-4)')
    parser.add_argument('--end_stage', type=int, default=5,
                       help='Which stage to end experiments at (1-5)')
    
    # Optimized ResNet50 specific parameters
    parser.add_argument('--optimized_only', action='store_true',
                       help='Run only optimized ResNet50 experiments')
    parser.add_argument('--comparison_only', action='store_true',
                       help='Run only comparison experiments between original and optimized ResNet50')
    parser.add_argument('--optimized_batch_size', type=int, default=16,
                       help='Batch size specific to optimized ResNet50')
    parser.add_argument('--optimized_lr', type=float, default=8e-5,
                       help='Learning rate specific to optimized ResNet50')
    
    return parser.parse_args()

def create_config(args, model_type, use_attention, scheduler_type):
    """Create configuration file"""
    config = {
        'data_dir': args.data_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'model_type': model_type,
        'use_attention': use_attention,
        'scheduler_type': scheduler_type
    }
    
    # Adjust parameters for optimized ResNet50
    if model_type == 'resnet50_optimized':
        # Use dedicated parameters or default optimized configuration
        optimized_batch_size = getattr(args, 'optimized_batch_size', args.batch_size // 2)
        optimized_lr = getattr(args, 'optimized_lr', args.learning_rate * 0.8)
        
        config.update({
            'batch_size': max(8, optimized_batch_size),   # Use dedicated batch size
            'learning_rate': optimized_lr,                # Use dedicated learning rate
            'weight_decay': args.weight_decay * 1.2,      # Increase regularization
            'patience': args.patience + 5,                # Increase patience value
            'gradient_clip': 1.0,                         # Add gradient clipping
            'accumulation_steps': 2,                      # Add gradient accumulation
            # Optimized version specific configuration
            'use_hard_mining': True,                      # Enable hard example mining
            'use_triangular_loss': True,                  # Enable triangular loss
            'mining_ratio': 0.25,                        # Hard example ratio
            'triangular_margin': 0.8,                    # Triangular loss margin
        })
    
    if scheduler_type == 'cosine':
        config.update({
            'cosine_T_max': args.num_epochs,
            'cosine_eta_min': 1e-6,
            'cosine_warmup_epochs': 10
        })
    
    # Create experiment ID
    experiment_id = f"{model_type}"
    if not use_attention:
        experiment_id += "_no_attention"
    experiment_id += f"_{scheduler_type}"
    experiment_id += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save configuration
    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/{experiment_id}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    
    return config_path, experiment_id

def run_experiment(config_path, experiment_id):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_id}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = f"python src/train_ensemble.py --config {config_path}"
    
    # Run command
    try:
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        
        print(f"\n{'='*80}")
        print(f"Experiment {experiment_id} completed")
        print(f"{'='*80}\n")
        return True
    except Exception as e:
        print(f"Error occurred while running experiment {experiment_id}: {str(e)}")
        return False

def get_experiment_stages(basic_models):
    """Get configuration for all experiment stages"""
    # Add ensemble to all models list
    all_models = basic_models + ['ensemble']
    
    # Check if optimized ResNet50 is included
    has_optimized_resnet = 'resnet50_optimized' in basic_models
    
    stages = [
        # Stage 1: All models (no attention) + cosine scheduler
        {
            'name': "All models (no attention) + cosine scheduler",
            'description': "Test performance of all models without attention mechanism, using simple cosine learning rate scheduler",
            'configs': [(model, False, 'cosine') for model in all_models]
        },
        # Stage 2: All models (no attention) + multi-stage scheduler
        {
            'name': "All models (no attention) + multi-stage scheduler",
            'description': "Test impact of multi-stage learning rate scheduler on all models",
            'configs': [(model, False, 'multistage') for model in all_models]
        },
        # Stage 3: All models (with attention) + cosine scheduler
        {
            'name': "All models (with attention) + cosine scheduler",
            'description': "Test impact of attention mechanism under cosine scheduler",
            'configs': [(model, True, 'cosine') for model in all_models]
        },
        # Stage 4: All models (with attention) + multi-stage scheduler
        {
            'name': "All models (with attention) + multi-stage scheduler",
            'description': "Test impact of attention mechanism under multi-stage scheduler",
            'configs': [(model, True, 'multistage') for model in all_models]
        }
    ]
    
    # If optimized ResNet50 is included, add dedicated comparison stage
    if has_optimized_resnet:
        stages.append({
            'name': "ResNet50 optimized version comparison",
            'description': "Directly compare performance of original ResNet50 vs optimized ResNet50 under different configurations",
            'configs': [
                ('resnet50', True, 'cosine'),
                ('resnet50_optimized', True, 'cosine'),
                ('resnet50', True, 'multistage'),
                ('resnet50_optimized', True, 'multistage'),
                ('resnet50', False, 'cosine'),
                ('resnet50_optimized', False, 'cosine'),
            ]
        })
    
    return stages

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create experiment log directory
    os.makedirs('experiment_logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_log = f"experiment_logs/systematic_comparison_{timestamp}.txt"
    experiment_summary = f"experiment_logs/summary_{timestamp}.txt"
    
    # Adjust model list based on run mode
    if args.optimized_only:
        if 'resnet50_optimized' not in args.basic_models:
            print("Error: Using --optimized_only mode but resnet50_optimized is not in basic models list")
            return
        args.basic_models = ['resnet50_optimized']
        print("Run mode: Optimized ResNet50 only")
    elif args.comparison_only:
        if 'resnet50_optimized' not in args.basic_models or 'resnet50' not in args.basic_models:
            print("Error: Using --comparison_only mode but resnet50 or resnet50_optimized is missing from basic models list")
            return
        args.basic_models = ['resnet50', 'resnet50_optimized']
        print("Run mode: ResNet50 comparison experiments only")
    
    # Get experiment stage configurations
    stages = get_experiment_stages(args.basic_models)
    
    # Filter stages to run
    stages_to_run = []
    for i, stage in enumerate(stages):
        stage_num = i + 1
        
        # Check if within range and not in skip list
        if (args.start_stage <= stage_num <= args.end_stage and 
            str(stage_num) not in args.skip_stages):
            stages_to_run.append((stage_num, stage))
    
    # Record all experiments to run
    all_experiments = []
    for stage_num, stage in stages_to_run:
        for model_type, use_attention, scheduler_type in stage['configs']:
            config_path, experiment_id = create_config(
                args, model_type, use_attention, scheduler_type
            )
            all_experiments.append({
                'stage': stage_num,
                'stage_name': stage['name'],
                'config_path': config_path,
                'experiment_id': experiment_id,
                'model_type': model_type,
                'use_attention': use_attention,
                'scheduler_type': scheduler_type
            })
    
    # Print experiment plan
    total_experiments = len(all_experiments)
    print(f"\n{'='*80}")
    print(f"Systematic comparison experiment plan - Total {total_experiments} experiments, {len(stages_to_run)} stages")
    print(f"{'='*80}")
    
    # Display optimized ResNet50 special configuration
    optimized_configs = [exp for exp in all_experiments if exp['model_type'] == 'resnet50_optimized']
    if optimized_configs:
        print(f"\nOptimized ResNet50 special configuration:")
        with open(optimized_configs[0]['config_path'], 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"  - Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  - Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"  - Weight decay: {config.get('weight_decay', 'N/A')}")
        print(f"  - Gradient clipping: {config.get('gradient_clip', 'N/A')}")
        print(f"  - Gradient accumulation steps: {config.get('accumulation_steps', 'N/A')}")
        print(f"  - Hard example mining: {config.get('use_hard_mining', 'N/A')}")
        print(f"  - Triangular loss: {config.get('use_triangular_loss', 'N/A')}")
        print(f"  - Hard example ratio: {config.get('mining_ratio', 'N/A')}")
        print(f"  - Triangular loss margin: {config.get('triangular_margin', 'N/A')}")
    
    for stage_num, stage in stages_to_run:
        print(f"\nStage {stage_num}: {stage['name']}")
        print(f"  Description: {stage['description']}")
        stage_experiments = [e for e in all_experiments if e['stage'] == stage_num]
        for i, exp in enumerate(stage_experiments):
            attention_str = "with attention" if exp['use_attention'] else "no attention"
            model_display = exp['model_type']
            if exp['model_type'] == 'resnet50_optimized':
                model_display += " (optimized)"
            print(f"  {i+1}. {model_display} ({attention_str}, {exp['scheduler_type']} scheduler)")
    
    print(f"\n{'='*80}")
    
    # Record experiment plan
    with open(experiment_log, 'w', encoding='utf-8') as f:
        f.write(f"Systematic comparison experiment plan - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total {total_experiments} experiments, {len(stages_to_run)} stages\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Training epochs: {args.num_epochs}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Early stopping patience: {args.patience}\n\n")
        
        for stage_num, stage in stages_to_run:
            f.write(f"\nStage {stage_num}: {stage['name']}\n")
            f.write(f"  Description: {stage['description']}\n")
            stage_experiments = [e for e in all_experiments if e['stage'] == stage_num]
            for i, exp in enumerate(stage_experiments):
                attention_str = "with attention" if exp['use_attention'] else "no attention"
                f.write(f"  {i+1}. {exp['model_type']} ({attention_str}, {exp['scheduler_type']} scheduler) - {exp['experiment_id']}\n")
            f.write("\n")
        
        f.write(f"\n{'='*80}\n\n")
    
    # Initialize results summary file
    with open(experiment_summary, 'w', encoding='utf-8') as f:
        f.write(f"Systematic comparison experiment results summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write("| Stage | Model | Attention | Scheduler | Accuracy(%) | Macro F1 | Weighted F1 | Duration(min) |\n")
        f.write("|-------|-------|-----------|------------|-------------|----------|-------------|---------------|\n")
    
    # Run all experiments
    current_stage = None
    
    for i, experiment in enumerate(all_experiments):
        # Check if entering new stage
        if current_stage != experiment['stage']:
            current_stage = experiment['stage']
            print(f"\n{'*'*80}")
            print(f"Starting stage {current_stage}: {experiment['stage_name']}")
            print(f"{'*'*80}\n")
        
        print(f"\nStarting experiment {i+1}/{total_experiments}")
        print(f"Model: {experiment['model_type']}, Attention: {'Yes' if experiment['use_attention'] else 'No'}, Scheduler: {experiment['scheduler_type']}")
        
        # Record start time
        start_time = datetime.now()
        
        # Run experiment
        success = run_experiment(experiment['config_path'], experiment['experiment_id'])
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0  # Convert to minutes
        
        # Try to get results from checkpoint
        accuracy = "N/A"
        macro_f1 = "N/A"
        weighted_f1 = "N/A"
        
        checkpoint_path = f"checkpoints/{experiment['experiment_id'].split('_')[0]}"
        if not experiment['use_attention']:
            checkpoint_path += "_no_attention"
        checkpoint_path += f"_{experiment['scheduler_type']}/best_model.pth"
        
        try:
            if os.path.exists(checkpoint_path):
                import torch
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                if 'val_acc' in checkpoint:
                    accuracy = f"{checkpoint['val_acc']*100:.2f}"
                if 'macro_f1' in checkpoint:
                    macro_f1 = f"{checkpoint['macro_f1']:.4f}"
                if 'weighted_f1' in checkpoint:
                    weighted_f1 = f"{checkpoint['weighted_f1']:.4f}"
        except Exception as e:
            print(f"Error reading checkpoint results: {str(e)}")
        
        # Record experiment results to log
        with open(experiment_log, 'a', encoding='utf-8') as f:
            result = "Success" if success else "Failed"
            f.write(f"Experiment {i+1}: {experiment['experiment_id']} - {result}\n")
            f.write(f"Model: {experiment['model_type']}, Attention: {'Yes' if experiment['use_attention'] else 'No'}, Scheduler: {experiment['scheduler_type']}\n")
            f.write(f"Accuracy: {accuracy}%, Macro F1: {macro_f1}, Weighted F1: {weighted_f1}\n")
            f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.2f} minutes\n\n")
        
        # Update summary table
        with open(experiment_summary, 'a', encoding='utf-8') as f:
            attention_str = "Yes" if experiment['use_attention'] else "No"
            f.write(f"| {experiment['stage']} | {experiment['model_type']} | {attention_str} | {experiment['scheduler_type']} | {accuracy} | {macro_f1} | {weighted_f1} | {duration:.2f} |\n")
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Experiment log saved to: {experiment_log}")
    print(f"Experiment results summary saved to: {experiment_summary}")
    print(f"{'='*80}")
    
    # Provide follow-up analysis suggestions
    print(f"\nFollow-up analysis suggestions:")
    print(f"1. View experiment results summary table: cat {experiment_summary}")
    print(f"2. Generate performance comparison report: python generate_report.py")
    print(f"3. Visualize training curves: python visualize_performance.py")
    
    if any(exp['model_type'] == 'resnet50_optimized' for exp in all_experiments):
        print(f"4. Analyze optimized ResNet50 improvement effects:")
        print(f"   - View hard example mining effects")
        print(f"   - Analyze multi-scale attention contribution")
        print(f"   - Compare triangular loss impact")
    
    print(f"\nQuickly reproduce best results:")
    print(f"Retrain with best performing configuration:")
    print(f"python src/train_ensemble.py --config configs/best_config.json")
    
    print(f"\nPerformance comparison summary:")
    print(f"Check results table in {experiment_summary} file")
    print(f"Focus on accuracy, macro F1, and weighted F1 metrics")
    
    if args.comparison_only:
        print(f"\nResNet50 comparison analysis:")
        print(f"Performance difference between original ResNet50 vs optimized ResNet50 has been recorded")
        print(f"Can focus on the effects of the following improvements:")
        print(f"- Impact of multi-scale residual attention")
        print(f"- Effects of hard example mining")  
        print(f"- Optimization of inter-class distance by triangular loss")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

"""
=============================================================================
Optimized ResNet50 Integration Usage Guide
=============================================================================

The script has successfully integrated the optimized ResNet50 model and supports the following run modes:

1. Complete comparison experiments (recommended):
   python run_comparison_experiments.py --data_dir /path/to/data

2. Test only optimized ResNet50:
   python run_comparison_experiments.py --optimized_only --data_dir /path/to/data

3. Original ResNet50 vs optimized version comparison:
   python run_comparison_experiments.py --comparison_only --data_dir /path/to/data

4. Custom optimized version parameters:
   python run_comparison_experiments.py --optimized_only \
     --optimized_batch_size 12 --optimized_lr 5e-5 --data_dir /path/to/data

5. Run specific stage:
   python run_comparison_experiments.py --start_stage 5 --end_stage 5 --data_dir /path/to/data

Special features of optimized ResNet50:
- Multi-scale residual attention mechanism
- Hard example mining loss
- Triangular loss optimization
- Adaptive weight adjustment
- Deep supervision training

Expected performance improvements:
- Validation accuracy: +2-4%
- Macro F1: +2-4%
- Hard example recognition: Significant improvement
- Class balance: Noticeable improvement

Results analysis:
- View summary table: cat experiment_logs/summary_*.txt
- View detailed log: cat experiment_logs/systematic_comparison_*.txt
- Focus on accuracy, macro F1, and weighted F1 metrics

Troubleshooting:
- Insufficient memory: Reduce --optimized_batch_size
- Unstable training: Lower --optimized_lr
- Time constraints: Use --comparison_only mode

=============================================================================
""" 