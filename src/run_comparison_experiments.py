import os
import subprocess
import argparse
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='对比实验脚本 - 系统化运行不同模型配置的对比实验')

    parser.add_argument('--data_dir', type=str, default='processed_data', help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')

    parser.add_argument('--basic_models', nargs='+', 
                       default=['resnet50', 'resnet50_optimized', 'efficientnet_b4', 'inceptionv3'],
                       help='要运行的基础模型列表')
    parser.add_argument('--skip_stages', nargs='+', default=[],
                       help='要跳过的实验阶段，从1到4')
    parser.add_argument('--start_stage', type=int, default=1,
                       help='从哪个阶段开始实验(1-4)')
    parser.add_argument('--end_stage', type=int, default=5,
                       help='在哪个阶段结束实验(1-5)')

    parser.add_argument('--optimized_only', action='store_true',
                       help='只运行优化版ResNet50的实验')
    parser.add_argument('--comparison_only', action='store_true',
                       help='只运行ResNet50原版与优化版的对比实验')
    parser.add_argument('--optimized_batch_size', type=int, default=16,
                       help='优化版ResNet50专用的批次大小')
    parser.add_argument('--optimized_lr', type=float, default=8e-5,
                       help='优化版ResNet50专用的学习率')

    return parser.parse_args()

def create_config(args, model_type, use_attention, scheduler_type):
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

    if model_type == 'resnet50_optimized':

        optimized_batch_size = getattr(args, 'optimized_batch_size', args.batch_size // 2)
        optimized_lr = getattr(args, 'optimized_lr', args.learning_rate * 0.8)

        config.update({
            'batch_size': max(8, optimized_batch_size),
            'learning_rate': optimized_lr,
            'weight_decay': args.weight_decay * 1.2,
            'patience': args.patience + 5,
            'gradient_clip': 1.0,
            'accumulation_steps': 2,

            'use_hard_mining': True,
            'use_triangular_loss': True,
            'mining_ratio': 0.25,
            'triangular_margin': 0.8,
        })

    if scheduler_type == 'cosine':
        config.update({
            'cosine_T_max': args.num_epochs,
            'cosine_eta_min': 1e-6,
            'cosine_warmup_epochs': 10
        })

    experiment_id = f"{model_type}"
    if not use_attention:
        experiment_id += "_no_attention"
    experiment_id += f"_{scheduler_type}"
    experiment_id += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/{experiment_id}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return config_path, experiment_id

def run_experiment(config_path, experiment_id):
    print(f"\n{'='*80}")
    print(f"开始运行实验: {experiment_id}")
    print(f"{'='*80}\n")

    cmd = f"python src/train_ensemble.py --config {config_path}"

    try:
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        print(f"\n{'='*80}")
        print(f"实验 {experiment_id} 已完成")
        print(f"{'='*80}\n")
        return True
    except Exception as e:
        print(f"运行实验 {experiment_id} 时发生错误: {str(e)}")
        return False

def get_experiment_stages(basic_models):

    all_models = basic_models + ['ensemble']

    has_optimized_resnet = 'resnet50_optimized' in basic_models

    stages = [

        {
            'name': "所有模型(无注意力)+余弦调度器",
            'description': "测试所有模型性能，不使用注意力机制，使用简单的余弦学习率调度器",
            'configs': [(model, False, 'cosine') for model in all_models]
        },

        {
            'name': "所有模型(无注意力)+多阶段调度器",
            'description': "测试多阶段学习率调度器对所有模型的影响",
            'configs': [(model, False, 'multistage') for model in all_models]
        },

        {
            'name': "所有模型(有注意力)+余弦调度器",
            'description': "测试注意力机制在余弦调度器下的影响",
            'configs': [(model, True, 'cosine') for model in all_models]
        },

        {
            'name': "所有模型(有注意力)+多阶段调度器",
            'description': "测试注意力机制在多阶段调度器下的影响",
            'configs': [(model, True, 'multistage') for model in all_models]
        }
    ]

    if has_optimized_resnet:
        stages.append({
            'name': "ResNet50优化版本对比",
            'description': "直接对比原始ResNet50与优化版ResNet50在不同配置下的性能",
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

    args = parse_args()

    os.makedirs('experiment_logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_log = f"experiment_logs/systematic_comparison_{timestamp}.txt"
    experiment_summary = f"experiment_logs/summary_{timestamp}.txt"

    if args.optimized_only:
        if 'resnet50_optimized' not in args.basic_models:
            print("错误：使用--optimized_only模式但基础模型列表中没有resnet50_optimized")
            return
        args.basic_models = ['resnet50_optimized']
        print("运行模式：仅优化版ResNet50")
    elif args.comparison_only:
        if 'resnet50_optimized' not in args.basic_models or 'resnet50' not in args.basic_models:
            print("错误：使用--comparison_only模式但基础模型列表中缺少resnet50或resnet50_optimized")
            return
        args.basic_models = ['resnet50', 'resnet50_optimized']
        print("运行模式：仅ResNet50对比实验")

    stages = get_experiment_stages(args.basic_models)

    stages_to_run = []
    for i, stage in enumerate(stages):
        stage_num = i + 1

        if (args.start_stage <= stage_num <= args.end_stage and 
            str(stage_num) not in args.skip_stages):
            stages_to_run.append((stage_num, stage))

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

    total_experiments = len(all_experiments)
    print(f"\n{'='*80}")
    print(f"系统化对比实验计划 - 总共 {total_experiments} 个实验, {len(stages_to_run)} 个阶段")
    print(f"{'='*80}")

    optimized_configs = [exp for exp in all_experiments if exp['model_type'] == 'resnet50_optimized']
    if optimized_configs:
        print(f"\n优化版ResNet50特殊配置:")
        with open(optimized_configs[0]['config_path'], 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"  - 批次大小: {config.get('batch_size', 'N/A')}")
        print(f"  - 学习率: {config.get('learning_rate', 'N/A')}")
        print(f"  - 权重衰减: {config.get('weight_decay', 'N/A')}")
        print(f"  - 梯度裁剪: {config.get('gradient_clip', 'N/A')}")
        print(f"  - 梯度累积步数: {config.get('accumulation_steps', 'N/A')}")
        print(f"  - 困难样本挖掘: {config.get('use_hard_mining', 'N/A')}")
        print(f"  - 三角损失: {config.get('use_triangular_loss', 'N/A')}")
        print(f"  - 困难样本比例: {config.get('mining_ratio', 'N/A')}")
        print(f"  - 三角损失边界: {config.get('triangular_margin', 'N/A')}")

    for stage_num, stage in stages_to_run:
        print(f"\n阶段 {stage_num}: {stage['name']}")
        print(f"  描述: {stage['description']}")
        stage_experiments = [e for e in all_experiments if e['stage'] == stage_num]
        for i, exp in enumerate(stage_experiments):
            attention_str = "有注意力" if exp['use_attention'] else "无注意力"
            model_display = exp['model_type']
            if exp['model_type'] == 'resnet50_optimized':
                model_display += " (优化版)"
            print(f"  {i+1}. {model_display} ({attention_str}, {exp['scheduler_type']}调度器)")

    print(f"\n{'='*80}")

    with open(experiment_log, 'w', encoding='utf-8') as f:
        f.write(f"系统化对比实验计划 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"总共 {total_experiments} 个实验, {len(stages_to_run)} 个阶段\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.num_epochs}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"权重衰减: {args.weight_decay}\n")
        f.write(f"早停耐心值: {args.patience}\n\n")

        for stage_num, stage in stages_to_run:
            f.write(f"\n阶段 {stage_num}: {stage['name']}\n")
            f.write(f"  描述: {stage['description']}\n")
            stage_experiments = [e for e in all_experiments if e['stage'] == stage_num]
            for i, exp in enumerate(stage_experiments):
                attention_str = "有注意力" if exp['use_attention'] else "无注意力"
                f.write(f"  {i+1}. {exp['model_type']} ({attention_str}, {exp['scheduler_type']}调度器) - {exp['experiment_id']}\n")
            f.write("\n")

        f.write(f"\n{'='*80}\n\n")

    with open(experiment_summary, 'w', encoding='utf-8') as f:
        f.write(f"系统化对比实验结果汇总 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write("| 阶段 | 模型 | 注意力 | 调度器 | 准确率(%) | 宏平均F1 | 加权F1 | 耗时(分钟) |\n")
        f.write("|------|------|--------|--------|-----------|----------|---------|------------|\n")

    current_stage = None

    for i, experiment in enumerate(all_experiments):

        if current_stage != experiment['stage']:
            current_stage = experiment['stage']
            print(f"\n{'*'*80}")
            print(f"开始阶段 {current_stage}: {experiment['stage_name']}")
            print(f"{'*'*80}\n")

        print(f"\n开始实验 {i+1}/{total_experiments}")
        print(f"模型: {experiment['model_type']}, 注意力: {'是' if experiment['use_attention'] else '否'}, 调度器: {experiment['scheduler_type']}")

        start_time = datetime.now()

        success = run_experiment(experiment['config_path'], experiment['experiment_id'])

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0

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
            print(f"读取检查点结果时出错: {str(e)}")

        with open(experiment_log, 'a', encoding='utf-8') as f:
            result = "成功" if success else "失败"
            f.write(f"实验 {i+1}: {experiment['experiment_id']} - {result}\n")
            f.write(f"模型: {experiment['model_type']}, 注意力: {'是' if experiment['use_attention'] else '否'}, 调度器: {experiment['scheduler_type']}\n")
            f.write(f"准确率: {accuracy}%, 宏平均F1: {macro_f1}, 加权F1: {weighted_f1}\n")
            f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"耗时: {duration:.2f}分钟\n\n")

        with open(experiment_summary, 'a', encoding='utf-8') as f:
            attention_str = "是" if experiment['use_attention'] else "否"
            f.write(f"| {experiment['stage']} | {experiment['model_type']} | {attention_str} | {experiment['scheduler_type']} | {accuracy} | {macro_f1} | {weighted_f1} | {duration:.2f} |\n")

    print(f"\n{'='*80}")
    print(f"所有实验已完成！")
    print(f"实验日志已保存到: {experiment_log}")
    print(f"实验结果汇总已保存到: {experiment_summary}")
    print(f"{'='*80}")

    print(f"\n后续分析建议:")
    print(f"1. 查看实验结果汇总表: cat {experiment_summary}")
    print(f"2. 生成性能对比报告: python generate_report.py")
    print(f"3. 可视化训练曲线: python visualize_performance.py")

    if any(exp['model_type'] == 'resnet50_optimized' for exp in all_experiments):
        print(f"4. 分析优化版ResNet50的改进效果:")
        print(f"   - 查看困难样本挖掘效果")
        print(f"   - 分析多尺度注意力的贡献")
        print(f"   - 对比三角损失的影响")

    print(f"\n快速重现最佳结果:")
    print(f"使用性能最好的配置重新训练:")
    print(f"python src/train_ensemble.py --config configs/best_config.json")

    print(f"\n性能对比总结:")
    print(f"检查 {experiment_summary} 文件中的结果表格")
    print(f"重点关注准确率、宏平均F1和加权F1指标")

    if args.comparison_only:
        print(f"\nResNet50对比分析:")
        print(f"原始ResNet50 vs 优化版ResNet50的性能差异已记录")
        print(f"可以重点关注以下改进点的效果:")
        print(f"- 多尺度残差注意力的影响")
        print(f"- 困难样本挖掘的效果")  
        print(f"- 三角损失对类间距离的优化")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
