import numpy as np
import torch
import torch.optim as optim
from model import LightGCN
from tqdm import tqdm

# --- 1. 加载数据与模型初始化 ---
print("--- Loading Data and Initializing Model ---")
data = torch.load("data.pt", weights_only=False)
model = LightGCN(num_users=data.num_users, num_movies=data.num_movies)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 2. 采样与BPR损失 (核心训练逻辑) ---
def sample_bpr_batch(data, batch_size):
    """
    为BPR Loss采样一个批次的数据。
    返回: users, positive_items, negative_items (都是在device上的tensor)
    """
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

# --- 3. 评估函数 (占位符) ---
@torch.no_grad()
def evaluate(model, data, k=20):
    """
    (占位符) 评估模型在测试集上的表现
    """
    print("Evaluation function is a placeholder and will be fully implemented next week.")
    return 0.0, 0.0

# --- 4. 完整的训练循环 ---
print("\n--- Step 2: Starting Full Training Loop ---")
num_epochs = 50
batch_size = 4096
num_train_interactions = data.train_pos_edge_index.shape[1]
steps_per_epoch = num_train_interactions // batch_size

# 我们需要完整的、用于信息传播的训练图结构
train_edge_index_u_to_m = data.train_pos_edge_index
train_edge_index_m_to_u = torch.stack([train_edge_index_u_to_m[1], train_edge_index_u_to_m[0]], dim=0)
train_total_edge_index = torch.cat([train_edge_index_u_to_m, train_edge_index_u_to_m], dim=1).to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()

            users, pos_items, neg_items = sample_bpr_batch(data, batch_size)

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