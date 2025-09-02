import argparse
import torch
from config import Config, print_config, save_config
from model import LightGCN
from train import train_and_evaluate
import os

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LightGCN on MovieLens')
    
    # 基本参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4096, help='批次大小')
    parser.add_argument('--embed_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=3, help='LightGCN层数')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2正则化')
    
    # 路径参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--data_path', type=str, default='./data.pt', help='数据文件路径')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--best_model_path', type=str, default='./best_model.pt', help='最佳模型保存路径')
    
    # 训练控制
    parser.add_argument('--eval_freq', type=int, default=5, help='评估频率(每N个epoch)')
    parser.add_argument('--save_model_freq', type=int, default=10, help='保存模型频率(每N个epoch)')
    parser.add_argument('--early_stop', action='store_true', help='是否启用早停')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置对象
    config = Config()
    
    # 用命令行参数更新配置
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    
    # 打印配置信息
    print_config(config)
    
    # --- 新增：定义全局日志文件 ---
    log_file_path = os.path.join("runs", "grid_search_log.txt")
    os.makedirs("runs", exist_ok=True) # 确保runs目录存在
    
    # 创建检查点目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(config.result_dir, f"{config.experiment_name}_config.json")
    save_config(config, config_path)
    
    # 设置随机种子
    config.set_seed()
    
    # 加载数据
    print("Loading data...")
    data = torch.load(config.data_path, weights_only=False)
    
    # 创建模型
    print("Creating model...")
    model = LightGCN(
        num_users=data.num_users, 
        num_movies=data.num_movies,
        embed_dim=config.embed_dim,
        n_layers=config.n_layers
    )
    
    # 训练和评估
    train_and_evaluate(model, data, config, log_file_path)
    
    print("Training completed!")

if __name__ == "__main__":
    main()