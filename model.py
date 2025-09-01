import torch
import torch.nn as nn
# 我们不再需要手动计算degree或使用scatter_add
# 直接导入官方为LightGCN设计的卷积层
from torch_geometric.nn import LGConv


class LightGCN(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=64, n_layers=3):
        super().__init__()

        self.num_users = num_users
        self.num_movies = num_movies
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.num_nodes = num_users + num_movies

        # 初始嵌入层，它也是第0层的嵌入
        self.embedding = nn.Embedding(self.num_nodes, self.embed_dim)

        # --- 【核心修正】 ---
        # 我们不再手动写循环，而是创建n_layers个LGConv层。
        # LGConv层非常“轻”，它内部没有需要学习的权重矩阵，
        # 它唯一的作用就是执行带有正确归一化的邻域聚合。
        self.convs = nn.ModuleList([LGConv() for _ in range(self.n_layers)])
        # --------------------

        nn.init.normal_(self.embedding.weight, std=0.1)

        print("LightGCN model (Production Version) created!")
        print(f"Number of layers: {self.n_layers}")

    def get_all_embeddings(self, edge_index):
        """
        【修正版】使用LGConv层来实现多层信息传播。
        """
        # 获取第0层的嵌入（初始嵌入）
        e0 = self.embedding.weight
        embeddings = [e0]  # 用于存储每一层的嵌入结果

        # --- 【核心修正】 ---
        # 用一个简洁的循环来调用LGConv层
        x = e0
        for i in range(self.n_layers):
            # LGConv的forward方法会自动处理所有复杂的归一化和聚合
            # 我们只需要传入上一层的嵌入和edge_index即可
            x = self.convs[i](x, edge_index)
            embeddings.append(x)
        # --------------------

        # 将所有层的嵌入堆叠起来，然后取平均
        # 这是LightGCN的“多层嵌入融合”步骤
        final_embeddings = torch.mean(torch.stack(embeddings, dim=0), dim=0)

        return final_embeddings

    def forward(self, user_indices, movie_indices, edge_index):
        """
        这个函数现在变得更简洁，因为它依赖get_all_embeddings
        """
        all_embeddings = self.get_all_embeddings(edge_index)

        user_embs = all_embeddings[user_indices]
        # 【重要修正】movie_indices在这里应该是已经加上偏移量的最终索引
        # 我们将在train.py中确保这一点
        movie_embs = all_embeddings[movie_indices]

        scores = torch.sum(user_embs * movie_embs, dim=1)
        return scores