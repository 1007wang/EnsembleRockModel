"""
对比实验脚本 - 自动化运行不同模型配置的对比实验
按照系统化、渐进的方式进行对比，清晰展示每个组件的贡献

支持的模型：
- resnet50: 原始ResNet50模型
- resnet50_optimized: 优化版ResNet50模型（多尺度注意力+困难样本挖掘+三角损失）
- efficientnet_b4: EfficientNet-B4模型
- inceptionv3: InceptionV3模型
- ensemble: 融合模型

实验阶段：
1. 所有模型(无注意力)+余弦调度器
2. 所有模型(无注意力)+多阶段调度器  
3. 所有模型(有注意力)+余弦调度器
4. 所有模型(有注意力)+多阶段调度器
5. ResNet50优化版本对比（仅当包含resnet50_optimized时）

使用示例：
1. 运行所有模型的完整对比实验：
   python run_comparison_experiments.py --data_dir /path/to/data

2. 只测试优化版ResNet50：
   python run_comparison_experiments.py --optimized_only --data_dir /path/to/data

3. 只对比原始ResNet50与优化版：
   python run_comparison_experiments.py --comparison_only --data_dir /path/to/data

4. 运行特定阶段的实验：
   python run_comparison_experiments.py --start_stage 3 --end_stage 5 --data_dir /path/to/data

5. 跳过某些阶段：
   python run_comparison_experiments.py --skip_stages 1 2 --data_dir /path/to/data

6. 使用自定义参数运行优化版ResNet50：
   python run_comparison_experiments.py --optimized_only --optimized_batch_size 12 --optimized_lr 5e-5 --data_dir /path/to/data
"""
import os
import subprocess
import argparse
import json
from datetime import datetime

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对比实验脚本 - 系统化运行不同模型配置的对比实验')
    
    # 基础训练参数
    parser.add_argument('--data_dir', type=str, default='processed_data', help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    
    # 实验控制参数
    parser.add_argument('--basic_models', nargs='+', 
                       default=['resnet50', 'resnet50_optimized', 'efficientnet_b4', 'inceptionv3'],
                       help='要运行的基础模型列表')
    parser.add_argument('--skip_stages', nargs='+', default=[],
                       help='要跳过的实验阶段，从1到4')
    parser.add_argument('--start_stage', type=int, default=1,
                       help='从哪个阶段开始实验(1-4)')
    parser.add_argument('--end_stage', type=int, default=5,
                       help='在哪个阶段结束实验(1-5)')
    
    # 优化版ResNet50专用参数
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
    """创建配置文件"""
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
    
    # 为优化版ResNet50调整参数
    if model_type == 'resnet50_optimized':
        # 使用专用参数或默认优化配置
        optimized_batch_size = getattr(args, 'optimized_batch_size', args.batch_size // 2)
        optimized_lr = getattr(args, 'optimized_lr', args.learning_rate * 0.8)
        
        config.update({
            'batch_size': max(8, optimized_batch_size),   # 使用专用batch size
            'learning_rate': optimized_lr,                # 使用专用学习率
            'weight_decay': args.weight_decay * 1.2,      # 增加正则化
            'patience': args.patience + 5,                # 增加耐心值
            'gradient_clip': 1.0,                         # 添加梯度裁剪
            'accumulation_steps': 2,                      # 添加梯度累积
            # 优化版特有的配置
            'use_hard_mining': True,                      # 启用困难样本挖掘
            'use_triangular_loss': True,                  # 启用三角损失
            'mining_ratio': 0.25,                        # 困难样本比例
            'triangular_margin': 0.8,                    # 三角损失边界
        })
    
    if scheduler_type == 'cosine':
        config.update({
            'cosine_T_max': args.num_epochs,
            'cosine_eta_min': 1e-6,
            'cosine_warmup_epochs': 10
        })
    
    # 创建实验ID
    experiment_id = f"{model_type}"
    if not use_attention:
        experiment_id += "_no_attention"
    experiment_id += f"_{scheduler_type}"
    experiment_id += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 保存配置
    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/{experiment_id}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    
    return config_path, experiment_id

def run_experiment(config_path, experiment_id):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"开始运行实验: {experiment_id}")
    print(f"{'='*80}\n")
    
    # 构建命令
    cmd = f"python src/train_ensemble.py --config {config_path}"
    
    # 运行命令
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
    """获取所有实验阶段的配置"""
    # 将ensemble添加到所有模型列表中
    all_models = basic_models + ['ensemble']
    
    # 检查是否包含优化版ResNet50
    has_optimized_resnet = 'resnet50_optimized' in basic_models
    
    stages = [
        # 阶段1：所有模型(无注意力)+余弦调度器
        {
            'name': "所有模型(无注意力)+余弦调度器",
            'description': "测试所有模型性能，不使用注意力机制，使用简单的余弦学习率调度器",
            'configs': [(model, False, 'cosine') for model in all_models]
        },
        # 阶段2：所有模型(无注意力)+多阶段调度器
        {
            'name': "所有模型(无注意力)+多阶段调度器",
            'description': "测试多阶段学习率调度器对所有模型的影响",
            'configs': [(model, False, 'multistage') for model in all_models]
        },
        # 阶段3：所有模型(有注意力)+余弦调度器
        {
            'name': "所有模型(有注意力)+余弦调度器",
            'description': "测试注意力机制在余弦调度器下的影响",
            'configs': [(model, True, 'cosine') for model in all_models]
        },
        # 阶段4：所有模型(有注意力)+多阶段调度器
        {
            'name': "所有模型(有注意力)+多阶段调度器",
            'description': "测试注意力机制在多阶段调度器下的影响",
            'configs': [(model, True, 'multistage') for model in all_models]
        }
    ]
    
    # 如果包含优化版ResNet50，添加专门的比较阶段
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
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建实验记录目录
    os.makedirs('experiment_logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_log = f"experiment_logs/systematic_comparison_{timestamp}.txt"
    experiment_summary = f"experiment_logs/summary_{timestamp}.txt"
    
    # 根据运行模式调整模型列表
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
    
    # 获取实验阶段配置
    stages = get_experiment_stages(args.basic_models)
    
    # 筛选要运行的阶段
    stages_to_run = []
    for i, stage in enumerate(stages):
        stage_num = i + 1
        
        # 检查是否在范围内且不在跳过列表中
        if (args.start_stage <= stage_num <= args.end_stage and 
            str(stage_num) not in args.skip_stages):
            stages_to_run.append((stage_num, stage))
    
    # 记录所有要运行的实验
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
    
    # 打印实验计划
    total_experiments = len(all_experiments)
    print(f"\n{'='*80}")
    print(f"系统化对比实验计划 - 总共 {total_experiments} 个实验, {len(stages_to_run)} 个阶段")
    print(f"{'='*80}")
    
    # 显示优化版ResNet50的特殊配置
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
    
    # 记录实验计划
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
    
    # 初始化结果汇总文件
    with open(experiment_summary, 'w', encoding='utf-8') as f:
        f.write(f"系统化对比实验结果汇总 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write("| 阶段 | 模型 | 注意力 | 调度器 | 准确率(%) | 宏平均F1 | 加权F1 | 耗时(分钟) |\n")
        f.write("|------|------|--------|--------|-----------|----------|---------|------------|\n")
    
    # 运行所有实验
    current_stage = None
    
    for i, experiment in enumerate(all_experiments):
        # 检查是否进入新阶段
        if current_stage != experiment['stage']:
            current_stage = experiment['stage']
            print(f"\n{'*'*80}")
            print(f"开始阶段 {current_stage}: {experiment['stage_name']}")
            print(f"{'*'*80}\n")
        
        print(f"\n开始实验 {i+1}/{total_experiments}")
        print(f"模型: {experiment['model_type']}, 注意力: {'是' if experiment['use_attention'] else '否'}, 调度器: {experiment['scheduler_type']}")
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 运行实验
        success = run_experiment(experiment['config_path'], experiment['experiment_id'])
        
        # 计算耗时
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0  # 转换为分钟
        
        # 尝试从检查点获取结果
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
        
        # 记录实验结果到日志
        with open(experiment_log, 'a', encoding='utf-8') as f:
            result = "成功" if success else "失败"
            f.write(f"实验 {i+1}: {experiment['experiment_id']} - {result}\n")
            f.write(f"模型: {experiment['model_type']}, 注意力: {'是' if experiment['use_attention'] else '否'}, 调度器: {experiment['scheduler_type']}\n")
            f.write(f"准确率: {accuracy}%, 宏平均F1: {macro_f1}, 加权F1: {weighted_f1}\n")
            f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"耗时: {duration:.2f}分钟\n\n")
        
        # 更新汇总表格
        with open(experiment_summary, 'a', encoding='utf-8') as f:
            attention_str = "是" if experiment['use_attention'] else "否"
            f.write(f"| {experiment['stage']} | {experiment['model_type']} | {attention_str} | {experiment['scheduler_type']} | {accuracy} | {macro_f1} | {weighted_f1} | {duration:.2f} |\n")
    
    print(f"\n{'='*80}")
    print(f"所有实验已完成！")
    print(f"实验日志已保存到: {experiment_log}")
    print(f"实验结果汇总已保存到: {experiment_summary}")
    print(f"{'='*80}")
    
    # 提供后续分析建议
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

"""
=============================================================================
优化版ResNet50集成使用指南
=============================================================================

现在脚本已成功集成优化版ResNet50模型，支持以下运行模式：

1. 完整对比实验（推荐）:
   python run_comparison_experiments.py --data_dir /path/to/data

2. 只测试优化版ResNet50:
   python run_comparison_experiments.py --optimized_only --data_dir /path/to/data

3. ResNet50原版vs优化版对比:
   python run_comparison_experiments.py --comparison_only --data_dir /path/to/data

4. 自定义优化版参数:
   python run_comparison_experiments.py --optimized_only \
     --optimized_batch_size 12 --optimized_lr 5e-5 --data_dir /path/to/data

5. 运行特定阶段:
   python run_comparison_experiments.py --start_stage 5 --end_stage 5 --data_dir /path/to/data

优化版ResNet50的特殊功能:
- 多尺度残差注意力机制
- 困难样本挖掘损失
- 三角损失优化
- 自适应权重调整
- 深度监督训练

预期性能提升:
- 验证准确率: +2-4%
- 宏平均F1: +2-4%
- 困难样本识别: 显著提升
- 类别均衡性: 明显改善

结果分析:
- 查看汇总表: cat experiment_logs/summary_*.txt
- 查看详细日志: cat experiment_logs/systematic_comparison_*.txt
- 重点关注准确率、宏平均F1和加权F1指标

故障排除:
- 内存不足: 减小--optimized_batch_size
- 训练不稳定: 降低--optimized_lr
- 时间限制: 使用--comparison_only模式

=============================================================================
""" 