import argparse
import torch
import os
from model import LightGCN
# 导入evaluate.py中的评估函数
from evaluate import evaluate
import json
from datetime import datetime

def parse_args():
    """解析评估命令行参数"""
    parser = argparse.ArgumentParser(description='LightGCN Evaluation')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=3, help='LightGCN层数')
    
    # 路径参数
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data_path', type=str, default='./data.pt', help='数据文件路径')
    parser.add_argument('--output_path', type=str, default='./results/eval_results.json', help='结果输出路径')
    
    # 评估参数
    parser.add_argument('--k', type=int, default=20, help='推荐列表长度(K)')
    parser.add_argument('--batch_size', type=int, default=256, help='评估批次大小')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')
    
    return parser.parse_args()

def main():
    """评估主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, weights_only=False)
    data.to(device)
    
    # 创建模型架构
    print(f"Creating model with embed_dim={args.embed_dim}, n_layers={args.n_layers}")
    model = LightGCN(
        num_users=data.num_users, 
        num_movies=data.num_movies,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers
    )
    
    # 加载模型权重
    print(f"Loading model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # 评估模型
    print(f"Evaluating model with K={args.k}...")
    model.eval()
    eval_results = evaluate(model, data, k=args.k, batch_size=args.batch_size)
    
    # 增加额外的元数据
    eval_results['model_path'] = args.model_path
    eval_results['evaluation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_results['k'] = args.k
    eval_results['embed_dim'] = args.embed_dim
    eval_results['n_layers'] = args.n_layers
    
    # 打印结果
    print("\n=== Evaluation Results ===")
    print(f"Validation Recall@{args.k}: {eval_results['val_recall']:.4f}")
    print(f"Validation NDCG@{args.k}: {eval_results['val_ndcg']:.4f}")
    print(f"Test Recall@{args.k}: {eval_results['test_recall']:.4f}")
    print(f"Test NDCG@{args.k}: {eval_results['test_ndcg']:.4f}")
    
    # 保存结果到JSON文件
    with open(args.output_path, 'w') as f:
        json.dump(eval_results, f, indent=4)

    # 同时保存为txt格式
    txt_output_path = args.output_path.replace('.json', '.txt')
    with open(txt_output_path, 'w') as f:
        f.write(f"==== LightGCN Evaluation Results ====\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Time: {eval_results['evaluation_time']}\n")
        f.write(f"Evaluation Parameters: K={args.k}, embed_dim={args.embed_dim}, n_layers={args.n_layers}\n")
        f.write("\n----- Results -----\n")
        f.write(f"Validation Recall@{args.k}: {eval_results['val_recall']:.4f}\n")
        f.write(f"Validation NDCG@{args.k}: {eval_results['val_ndcg']:.4f}\n")
        f.write(f"Test Recall@{args.k}: {eval_results['test_recall']:.4f}\n")
        f.write(f"Test NDCG@{args.k}: {eval_results['test_ndcg']:.4f}\n")

    print(f"\nResults saved to {args.output_path} (JSON) and {txt_output_path} (TXT)")

if __name__ == "__main__":
    main()
