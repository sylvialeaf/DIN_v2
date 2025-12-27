#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立 DIN 项目 - 主运行入口

这是一个完全不依赖 RecBole 框架的 DIN 实现项目。
专为研究 DIN 模型的三个关键问题而设计。

实验目录:
- experiment1.py: 历史序列长度敏感性分析
- experiment2.py: DIN vs 传统方法对比
- experiment3.py: DIN 改进消融实验

使用方法:
    python run_experiments.py           # 运行所有实验
    python run_experiments.py 1         # 只运行实验一
    python run_experiments.py 2         # 只运行实验二
    python run_experiments.py 3         # 只运行实验三
    python run_experiments.py 1 2       # 运行实验一和二
"""

import os
import sys
import subprocess
from datetime import datetime

EXPERIMENTS = {
    '1': {
        'name': 'experiment1.py',
        'title': '实验一: 历史序列长度敏感性分析',
        'description': '测试 DIN、GRU4Rec、SASRec、NARM 在不同历史长度下的表现'
    },
    '2': {
        'name': 'experiment2.py',
        'title': '实验二: DIN vs 传统方法对比',
        'description': '对比 DIN、AvgPool、LightGBM、混合精排的效果和效率'
    },
    '3': {
        'name': 'experiment3.py',
        'title': '实验三: DIN 改进消融实验',
        'description': '测试时间衰减注意力和多头注意力的改进效果'
    },
    '4': {
        'name': 'experiment4.py',
        'title': '实验四: 高级改进探索',
        'description': '自适应时间衰减和对比学习预训练'
    }
}


def run_experiment(exp_id):
    """运行单个实验"""
    exp = EXPERIMENTS.get(str(exp_id))
    if not exp:
        print(f"❌ 未知实验: {exp_id}")
        return False
    
    script_path = os.path.join(os.path.dirname(__file__), exp['name'])
    
    if not os.path.exists(script_path):
        print(f"❌ 脚本不存在: {script_path}")
        return False
    
    print("\n" + "=" * 80)
    print(f"🔬 {exp['title']}")
    print(f"   {exp['description']}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验 {exp_id} 失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("🎯 独立 DIN 研究项目 - 丰富特征版")
    print("=" * 80)
    print("""
    这是一个完全独立于 RecBole 的序列推荐模型实现项目。
    无需复杂框架依赖，纯 PyTorch 实现，易于理解和修改。
    
    项目结构:
    ├── models.py              - DIN、GRU4Rec、SASRec、NARM、AvgPool
    ├── data_loader.py         - MovieLens 数据加载器 (含丰富特征)
    ├── feature_engineering.py - 特征工程模块
    ├── trainer.py             - 训练器（含早停、评估）
    ├── experiment1.py         - 序列长度 & 模型对比实验
    ├── experiment2.py         - DIN vs 传统方法对比
    └── experiment3.py         - 消融实验
    """)
    
    # 确定要运行的实验
    if len(sys.argv) > 1:
        experiments_to_run = sys.argv[1:]
    else:
        # 默认运行所有实验
        experiments_to_run = ['1', '2', '3', '4']
        print(f"\n📋 将运行所有实验 (带丰富特征)")
        print(f"   可选: 1, 2, 3, 4")
        print(f"   示例: python run_experiments.py 1 2")
    
    print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行实验
    results = {}
    for exp_id in experiments_to_run:
        success = run_experiment(exp_id)
        results[exp_id] = success
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 运行总结")
    print("=" * 80)
    
    for exp_id, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        exp = EXPERIMENTS.get(str(exp_id), {})
        print(f"   {status}: {exp.get('title', f'实验 {exp_id}')}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n   完成: {success_count}/{total_count} 个实验")
    print(f"   结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查结果文件
    results_dir = os.path.join(os.path.dirname(__file__), 'results_gpu')
    if os.path.exists(results_dir):
        result_files = os.listdir(results_dir)
        if result_files:
            print(f"\n📁 结果文件 ({results_dir}):")
            for f in sorted(result_files)[-10:]:  # 只显示最近10个
                print(f"   - {f}")
    
    print("=" * 80)
    
    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
