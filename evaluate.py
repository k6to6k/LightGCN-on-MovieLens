import torch
import numpy as np
from tqdm import tqdm
import math

# 获取用户的物品交互
@torch.no_grad()
def get_user_positive_items(edge_index):
    """
    为每个用户获取其正向交互的物品列表
    
    参数:
        edge_index: 形状为[2, num_edges]的边索引张量
                   
    返回:
        dict: {user_id: [positive_items]}，每个用户的正向交互物品列表
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0, i].item()
        item = edge_index[1, i].item()

        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)

    return user_pos_items

def calc_recall(topk_items, positive_items, k):
    """
    计算Recall@K
    
    参数:
        topk_items: 为用户推荐的Top-K物品列表
        positive_items: 用户在测试集中真实交互的物品列表
        k: 推荐列表长度
        
    返回:
        recall: 召回率
    """
    # 计算推荐列表中有多少是真实交互的物品
    hit_items = set(topk_items[:k]) & set(positive_items)
    recall = len(hit_items) / len(positive_items) if (len(positive_items) > 0) else 0
    return recall

def calc_ndcg(topk_items, positive_items, k):
    """
    计算NDCG@K
    
    参数:
        topk_items: 为用户推荐的Top-K物品列表
        positive_items: 用户在测试集中真实交互的物品列表
        k: 推荐列表长度
        
    返回:
        ndcg: NDCG@K 归一化折损累积增益
    """
    # 将TopK推荐列表中的每个物品标记为1（命中）或0（未命中）
    relevance = np.zeros(k)
    for i, item in enumerate(topk_items[:k]):
        if item in positive_items:
            relevance[i] = 1

    # 计算DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i in range(k):
        if relevance[i] > 0:
            # i+2是因为我们的推荐排名从0开始，但公式中从1开始
            # 使用对数底数为2的对数
            dcg += relevance[i] / math.log2(i + 2)

    # 计算IDCG (Ideal DCG)
    # IDCG是将所有相关物品（这里最多为k个）放在最前面得到的最大DCG
    n_rel = min(len(positive_items), k)
    idcg = 0.0
    for i in range(n_rel):
        idcg += 1.0 / math.log2(i + 2)
    
    # 计算NDCG@K
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg

@torch.no_grad()
def evaluate_model(model, data, split="val", k=20, batch_size=256):
    """
    评估模型在指定数据集上的表现
    
    参数:
        model: LightGCN模型实例
        data: 包含图数据的PyG Data对象
        split: 评估数据集，"val"表示验证集，"test"表示测试集
        k: 推荐列表长度
        batch_size: 批次大小
        
    返回:
        float, float: Recall@K和NDCG@K的平均值
    """
    model.eval() # 设置为评估模式
    device = next(model.parameters()).device

    # 获取用户在训练集和评估集(验证集或测试集)中交互的物品
    train_user_pos_items = get_user_positive_items(data.train_pos_edge_index)

    if split == "val":
        eval_edge_index = data.val_pos_edge_index
    else: # split == "test"
        eval_edge_index = data.test_pos_edge_index
    
    eval_user_pos_items = get_user_positive_items(eval_edge_index)

    # 计算所有最终嵌入
    all_embeddings = model.get_all_embeddings(data.edge_index.to(model.embedding.weight.device))

    # 获取所有用户和物品的嵌入
    user_embeddings = all_embeddings[:data.num_users]
    item_embeddings = all_embeddings[data.num_users:]

    # 为每个在评估集中有交互的用户计算指标
    recalls, ndcgs = [], []
    eval_users = list(eval_user_pos_items.keys())

    # 使用批处理评估，防止内存溢出
    num_eval_users = len(eval_users)
    num_batches = (num_eval_users + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_eval_users)
        batch_users = eval_users[start:end]

        batch_users_embeddings = user_embeddings[batch_users]

        # 计算这批用户与所有物品的评分 (用户嵌入 * 物品嵌入)
        # 结果形状: [batch_size, num_items]
        scores = torch.matmul(batch_users_embeddings, item_embeddings.t()).cpu().numpy()

        # 对每个用户计算top-k推荐列表
        for i, user in enumerate(batch_users):
            user_scores = scores[i]

            # 排除训练集中已交互的物品
            if user in train_user_pos_items:
                train_items = train_user_pos_items[user]
                # 将训练集中的物品得分嗯设为负无穷，使它们不会出现在推荐列表当中
                for item in train_items:
                    # 注意物品ID需要减去用户数量的偏移
                    user_scores[item - data.num_users] = float('-inf')

            # 按得分排序获取物品索引（从高到低）
            topk_items_indices = np.argsort(-user_scores)[:k]
            # 加上偏移量转换为原始物品ID
            topk_items = [idx + data.num_users for idx in topk_items_indices]

            # 计算评估指标
            recall = calc_recall(topk_items, eval_user_pos_items[user], k)
            ndcg = calc_ndcg(topk_items, eval_user_pos_items[user], k)

            recalls.append(recall)
            ndcgs.append(ndcg)

    # 计算所有用户的平均 recall 和 ndcg
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0

    return avg_recall, avg_ndcg
    
@torch.no_grad()
def evaluate(model, data, k=20, batch_size=256):
    """
    评估模型在测试集上的表现
    
    参数:
        model: LightGCN模型实例
        data: 包含图数据的PyG Data对象
        k: 推荐列表长度
        batch_size: 批次大小
    
    返回:
        dict: 包含验证集和测试集上的Recall@K和NDCG@K
    """
    # 评估验证集
    val_recall, val_ndcg = evaluate_model(model, data, split="val", k=k, batch_size=batch_size)
    
    # 评估测试集
    test_recall, test_ndcg = evaluate_model(model, data, split="test", k=k, batch_size=batch_size)
    
    results = {
        "val_recall": val_recall,
        "val_ndcg": val_ndcg,
        "test_recall": test_recall,
        "test_ndcg": test_ndcg
    }
    
    return results