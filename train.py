import numpy as np
import torch
import torch.optim as optim
from model import LightGCN
from tqdm import tqdm
import os
from evaluate import evaluate as evaluate_function
from datetime import datetime

# --- 2. 采样与BPR损失 (核心训练逻辑) ---
def sample_bpr_batch(data, batch_size, device=None):
    """
    为BPR Loss采样一个批次的数据。
    返回: users, positive_items, negative_items (都是在device上的tensor)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    train_interactions = data.train_pos_edge_index
    num_interactions = train_interactions.shape[1]

    # 随机采样一批交互的索引
    sample_indices = torch.randint(0, num_interactions, (batch_size,)).to(device)

    # 提取对应的用户和正样本电影
    users = train_interactions[0, sample_indices]
    pos_items = train_interactions[1, sample_indices]

    # 为每个用户采样一个负样本电影
    neg_items = torch.randint(data.num_users, data.num_users + data.num_movies, (batch_size,)).to(device)

    return users, pos_items, neg_items

def bpr_loss(pos_scores, neg_scores):
    return -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

def log_to_file(file_path, content):
    """追加内容到指定文件"""
    with open(file_path, 'a') as f:
        f.write(content + '\n')

# --- 3. 评估函数 (现在调用evaluate.py中的函数) ---
@torch.no_grad()
def evaluate(model, data, k=20):
    """
    评估模型在测试集上的表现，通过调用evaluate.py中的函数
    """
    # 调用evaluate.py中的evaluate函数
    results = evaluate_function(model, data, k=k)
    
    # 返回与原接口相同的结果 (recall, ndcg)
    return results['val_recall'], results['val_ndcg']

# --- 添加train_and_evaluate函数 ---
def train_and_evaluate(model, data, config, log_file_path=None):
    """
    训练模型并定期评估，供main.py调用
    
    参数:
        model: LightGCN模型实例
        data: 包含图数据的PyG Data对象
        config: 配置对象
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
    model.to(device)
    data.to(device)
    
    # --- 新增：在日志文件中记录本次实验的开始 ---
    if log_file_path:
        log_content = f"\n===== Experiment Start: wd={config.weight_decay}, lr={config.lr} ====="
        log_to_file(log_file_path, log_content)
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr, 
                         weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.0)
    
    # 初始化训练参数
    num_epochs = config.epochs
    batch_size = config.batch_size
    num_train_interactions = data.train_pos_edge_index.shape[1]
    steps_per_epoch = num_train_interactions // batch_size
    
    # 初始化早停相关变量
    best_val_metric = 0
    best_epoch = 0
    patience_counter = 0
    
    # 我们需要完整的、用于信息传播的训练图结构
    train_edge_index_u_to_m = data.train_pos_edge_index
    train_edge_index_m_to_u = torch.stack([train_edge_index_u_to_m[1], train_edge_index_u_to_m[0]], dim=0)
    train_total_edge_index = torch.cat([train_edge_index_u_to_m, train_edge_index_m_to_u], dim=1).to(device)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for _ in range(steps_per_epoch):
                optimizer.zero_grad()
                
                users, pos_items, neg_items = sample_bpr_batch(data, batch_size, device)
                
                all_final_embeddings = model.get_all_embeddings(train_total_edge_index)
                
                user_embs = all_final_embeddings[users]
                pos_item_embs = all_final_embeddings[pos_items]
                neg_item_embs = all_final_embeddings[neg_items]
                
                pos_scores = torch.sum(user_embs * pos_item_embs, dim=1)
                neg_scores = torch.sum(user_embs * neg_item_embs, dim=1)
                
                loss = bpr_loss(pos_scores, neg_scores)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        
        avg_loss = total_loss / steps_per_epoch
        print(f"\nEpoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
        
        # 定期评估
        if (epoch + 1) % config.eval_freq == 0:
            model.eval()
            # 使用我们新的评估模块
            eval_results = evaluate_function(model, data, k=config.k if hasattr(config, 'k') else 20)
            
            val_recall = eval_results['val_recall']
            val_ndcg = eval_results['val_ndcg']
            test_recall = eval_results['test_recall']
            test_ndcg = eval_results['test_ndcg']

            print(f"--- Validation at Epoch {epoch + 1} ---")
            print(f"Recall@{config.k if hasattr(config, 'k') else 20}: {val_recall:.4f}, "
                  f"NDCG@{config.k if hasattr(config, 'k') else 20}: {val_ndcg:.4f}")
            print(f"Test Recall@{config.k if hasattr(config, 'k') else 20}: {test_recall:.4f}, "
                  f"Test NDCG@{config.k if hasattr(config, 'k') else 20}: {test_ndcg:.4f}")
            
            # --- 新增：将评估结果写入日志文件 ---
            if log_file_path:
                log_content = (f"Epoch {epoch + 1:3d} | "
                               f"Val Recall@{config.k or 20}: {val_recall:.4f}, Val NDCG@{config.k or 20}: {val_ndcg:.4f} | "
                               f"Test Recall@{config.k or 20}: {test_recall:.4f}, Test NDCG@{config.k or 20}: {test_ndcg:.4f}")
                log_to_file(log_file_path, log_content)

            # 使用验证集指标进行早停检查
            current_metric = val_recall
            
            # 保存检查点
            if hasattr(config, 'save_model_freq') and (epoch + 1) % config.save_model_freq == 0:
                checkpoint_dir = config.checkpoint_dir if hasattr(config, 'checkpoint_dir') else "./checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # 检查是否为最佳模型
            min_delta = config.min_delta if hasattr(config, 'min_delta') else 0.001
            if current_metric > best_val_metric + min_delta:
                best_val_metric = current_metric
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存最佳模型
                best_model_path = config.best_model_path if hasattr(config, 'best_model_path') else "./best_model.pt"
                os.makedirs(os.path.dirname(best_model_path) if os.path.dirname(best_model_path) else ".", exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved! Val Recall@{config.k if hasattr(config, 'k') else 20}: {best_val_metric:.4f}")
            else:
                patience_counter += 1
                if hasattr(config, 'early_stop_patience'):
                    print(f"No improvement. Patience: {patience_counter}/{config.early_stop_patience}")
            
            # 早停检查 (修正逻辑)
            if config.early_stop and config.early_stop_patience > 0 and patience_counter >= config.early_stop_patience:
                print(f"Early stopping triggered! No improvement for {patience_counter} evaluations.")
                print(f"Best epoch was {best_epoch} with Val Recall@{config.k if hasattr(config, 'k') else 20}: {best_val_metric:.4f}")
                
                # --- 新增：在日志中记录早停事件 ---
                if log_file_path:
                    log_to_file(log_file_path, f"Early stopping at Epoch {epoch + 1}. Best Epoch: {best_epoch}, Best Val Recall: {best_val_metric:.4f}")
                
                break
    
    # 保存最终模型（保持原有行为）
    print("\n--- Training Finished ---")
    torch.save(model.state_dict(), 'lightgcn_model.pt')
    print("Final model state_dict saved to lightgcn_model.pt")
    
    # 如果存在最佳模型，加载并进行最终评估
    best_model_path = config.best_model_path if hasattr(config, 'best_model_path') else "./best_model.pt"
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} (achieved at Epoch {best_epoch})")
        model.load_state_dict(torch.load(best_model_path))
        
        model.eval()
        final_results = evaluate_function(model, data)
        print("\n=== Final Evaluation Results ===")
        print(f"Test Recall@{config.k if hasattr(config, 'k') else 20}: {final_results['test_recall']:.4f}")
        print(f"Test NDCG@{config.k if hasattr(config, 'k') else 20}: {final_results['test_ndcg']:.4f}")

        # --- 新增：将最终总结写入summary文件 ---
        summary_path = os.path.join(config.result_dir, "summary.txt")
        summary_content = (
            f"===== Experiment Summary =====\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Best Model Path: {best_model_path}\n"
            f"---- Hyperparameters ----\n"
            f"Learning Rate (lr): {config.lr}\n"
            f"Weight Decay (wd): {config.weight_decay}\n"
            f"Embed Dim: {config.embed_dim}\n"
            f"Num Layers: {config.n_layers}\n"
            f"---- Training Summary ----\n"
            f"Best Epoch: {best_epoch}\n"
            f"Best Validation Recall@{config.k or 20}: {best_val_metric:.4f}\n"
            f"---- Final Test Performance ----\n"
            f"Test Recall@{config.k or 20}: {final_results['test_recall']:.4f}\n"
            f"Test NDCG@{config.k or 20}: {final_results['test_ndcg']:.4f}\n"
            f"=============================="
        )
        with open(summary_path, 'w') as f:
            f.write(summary_content)

        # --- 新增：在日志中记录最终结果 ---
        if log_file_path:
            log_to_file(log_file_path, "--- Final Test Performance (on best model) ---")
            log_to_file(log_file_path, f"Test Recall@{config.k or 20}: {final_results['test_recall']:.4f}, Test NDCG@{config.k or 20}: {final_results['test_ndcg']:.4f}")
            log_to_file(log_file_path, "===== Experiment End =====")

    return model

# --- 保留直接运行能力 ---
if __name__ == "__main__":
    # --- 1. 加载数据与模型初始化 ---
    print("--- Loading Data and Initializing Model ---")
    data = torch.load("data.pt", weights_only=False)
    model = LightGCN(num_users=data.num_users, num_movies=data.num_movies)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # --- 4. 完整的训练循环 ---
    print("\n--- Step 2: Starting Full Training Loop ---")
    num_epochs = 50
    batch_size = 4096
    num_train_interactions = data.train_pos_edge_index.shape[1]
    steps_per_epoch = num_train_interactions // batch_size
    
    # 我们需要完整的、用于信息传播的训练图结构
    train_edge_index_u_to_m = data.train_pos_edge_index
    train_edge_index_m_to_u = torch.stack([train_edge_index_u_to_m[1], train_edge_index_u_to_m[0]], dim=0)
    train_total_edge_index = torch.cat([train_edge_index_u_to_m, train_edge_index_m_to_u], dim=1).to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for _ in range(steps_per_epoch):
                optimizer.zero_grad()
                
                users, pos_items, neg_items = sample_bpr_batch(data, batch_size, device)
                
                all_final_embeddings = model.get_all_embeddings(train_total_edge_index)
                
                user_embs = all_final_embeddings[users]
                pos_item_embs = all_final_embeddings[pos_items]
                neg_item_embs = all_final_embeddings[neg_items]
                
                pos_scores = torch.sum(user_embs * pos_item_embs, dim=1)
                neg_scores = torch.sum(user_embs * neg_item_embs, dim=1)
                
                loss = bpr_loss(pos_scores, neg_scores)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        
        avg_loss = total_loss / steps_per_epoch
        print(f"\nEpoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            recall, ndcg = evaluate(model, data, k=20)
            print(f"--- Validation at Epoch {epoch + 1} ---")
            print(f"Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
    
    print("\n--- Training Finished ---")
    
    # --- 保存最终模型 ---
    torch.save(model.state_dict(), 'lightgcn_model.pt')
    print("Final model state_dict saved to lightgcn_model.pt")