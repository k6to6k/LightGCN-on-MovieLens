import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from torch_scatter import scatter_add

class LightGCN(nn.Module):
    """
    LightGCN模型的核心实现。
    """
    def __init__(self, num_users, num_movies, embed_dim=64, n_layers=3):
        """
        模型的构造函数。

        参数:
        - num_users (int): 数据集中的用户总数。
        - num_movies (int): 数据集中的电影总数。
        - embed_dim (int): 嵌入向量的维度，一个重要的超参数。
        - n_layers(int): GCN的层数，即信息传播的次数
        """
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # -------------------- 核心组件定义 --------------------

        # 1. 创建一个巨大的Embedding层。
        #    它需要能够为每一个用户和每一个电影都提供一个嵌入向量。
        #    所以，它的总大小应该是 (用户数 + 电影数)。
        num_nodes = num_users + num_movies
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, embed_dim)

        # 2. (可选，但推荐) 为嵌入层提供一个较好的随机初始化。
        #    这有助于模型更快、更稳定地收敛。
        nn.init.xavier_uniform_(self.embedding.weight)

        print("LightGCN model skeleton created!")
        print(f"Total number of nodes (users + movies): {num_nodes}")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Number of layers: {self.n_layers}")


    def get_all_embeddings(self, edge_index):
        """
        实现多层信息传播的核心函数。
        """
        # 获取初始的、完整的嵌入矩阵
        all_embeddings = self.embedding.weight

        # 用于存储每一层传播后的嵌入结果
        # 我们需要把第0层的初始嵌入也存进去
        embeddings_list = [all_embeddings]

        # --- 手动实现传播循环 ---
        # LightGCN的传播公式本质是：E' = (D^-0.5 * A * D^-0.5) * E
        # 其中A是邻接矩阵，D是度矩阵。这需要复杂的稀疏矩阵运算。
        # 可以通过两步消息传递和归一化来实现

        # 1. 计算归一化系数 D^-0.5
        # row是边的起点，col是边的终点
        row, col = edge_index
        num_nodes = self.num_nodes + self.num_movies
        # 计算每个节点的度（入度+出度，因为我们的图是对称的）
        deg = degree(row, all_embeddings.size(0), dtype=all_embeddings.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        # 将无穷大的值（如果一个节点度为0）替换为0
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # 这就是每个节点的归一化系数
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 手动实现传播循环
        current_embeddings = all_embeddings
        for layer in range(self.n_layers):
            # 这是一个简化的邻域聚合，暂时不考虑归一化
            # 我们需要一种方法，把edge_index[1]（源节点）的特征，聚合到edge_index[0]（目标节点）上
            # 我们可以使用一个for循环，但这效率极低。
            # 一个高效的方法是使用scatter_add_，但这比较底层。

            # 【教学简化版】
            # 我们先用一个概念性的伪代码来理解，然后在完整版中给出高效实现
            # next_embeddings = aggregate_neighbors(current_embeddings, edge_index)
            # current_embeddings = next_embeddings

            # -------------------- 填空部分 --------------------
            # 从edge_index中获取源节点和目标节点
            source_nodes = edge_index[1]
            target_nodes = edge_index[0]

            # 从当前嵌入矩阵中，取出所有源节点的特征
            source_embeddings = current_embeddings[source_nodes]
            # next_embeddings = torch.zeros_like(current_embeddings)

            # 在传递消息前，先对消息应用归一化权重
            messages = source_embeddings * norm.view(-1, 1)

            '''
            真正的、深层的错误在于： 我之前指导您编写的这个手动传播循环
            虽然在概念上是正确的（加权、聚合），但它在具体实现上，与PyG官方的 LGConv 层相比
            缺少了很多内部的、必要的形状检查和处理逻辑
            scatter_add 是一个非常底层的工具，如果给它的输入（特别是索引）有任何问题，它可能会产生意想不到的形状输出。
            '''

            # 【核心优化】使用 torch_scatter.scatter_add 进行最高效的聚合
            aggregated_embeddings = scatter_add(messages, target_nodes, dim=0, dim_size=num_nodes)

            # 将源节点的特征“发送”给目标节点
            # 提示：使用 .index_add_() 方法可以高效地完成这个操作
            # next_embeddings.index_add_(0, target_nodes, current_embeddings[source_nodes])


            # 更新当前嵌入
            current_embeddings = aggregated_embeddings

            # 将新一层的嵌入结果存入列表
            embeddings_list.append(current_embeddings)

            # 将所有层的嵌入结果堆叠起来
            # 提示：使用 torch.stack
        final_embeddings_stack = torch.stack(embeddings_list, dim=0)

        # 对所有层的嵌入取平均，作为最终的嵌入
        final_embeddings = torch.mean(final_embeddings_stack, dim=0)

        return final_embeddings


    def forward(self, users_indices, movies_indices, edge_index):
        """
         前向传播的初版。
         目标：根据输入的用户和电影ID，从Embedding层中查找到对应的向量。

         参数:
         - user_indices (torch.Tensor): 一批用户的索引。
         - movie_indices (torch.Tensor): 一批电影的索引。
         - edge_index (torch.Tensor): 边的索引 [2, N] 张量
         """
        # -------------------- 基础嵌入查找 --------------------

        all_embeddings = self.get_all_embeddings(edge_index)

        # 1. 从Embedding层中查找用户的嵌入向量。
        # user_embeddings = self.embedding(users_indices)
        user_embs = all_embeddings[users_indices]

        # 2. 从Embedding层中查找电影的嵌入向量。
        #    【注意！】电影的真实索引需要加上用户总数的偏移量。
        #    但在这里，我们假设传入的movie_indices已经是处理过的最终索引了。
        # movie_embeddings = self.embedding(movies_indices)
        movie_embs = all_embeddings[movies_indices]
        scores = torch.sum(user_embs * movie_embs, dim=1)

        # 3. 返回查找到的嵌入向量
        return scores