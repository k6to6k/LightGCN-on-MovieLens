#%%
import pandas as pd
import numpy as np
import os
#%%
print("Step 1: Data Loading & Exploration")
#%%
# 定义数据文件的路径
# 使用os.path.join来构建跨平台兼容的路径
data_dir = './ml-1m'
ratings_file = os.path.join(data_dir, 'ratings.dat')
#%%
# MovieLens-1M 使用 ‘::' 为分隔符
# 文件没有标题，所以我们提供列名称
ratings_df = pd.read_csv(
    ratings_file,
    sep='::',
    header=None,
    names=['userId', 'movieId', 'rating', 'timestamp'],
    engine='python' # 必需，因为“::”不是标准的单字符分隔符
)
#%%
# 打印基本信息，检查数据类型和是否有缺失值
print("\nDataFrame Info:")
ratings_df.info()
#%%
# 打印描述性统计，了解数据的分布
print("\nDescriptive statistics:")
print(ratings_df.describe())
#%%
# 打印DataFrame的前5行，检查是否加载正确
print("First 5 rows of the dataset:")
print(ratings_df.head(5))
#%%
# 使用 .nunique() 方法计算独立用户和电影的数量
num_users = ratings_df['userId'].nunique()
num_movies = ratings_df['movieId'].nunique()
print(f"\nNumber of unique users: {num_users}")
print(f"Number of unique movies: {num_movies}")
#%%
# ID重映射
print("\n--- Step 2: ID Remapping ---")

# 从DataFrame中获取所有唯一的用户ID和电影ID
# 提示：使用 .unique() 方法
unique_user_id = ratings_df['userId'].unique()
unique_movie_id = ratings_df['movieId'].unique()

print("\nUnique users:")
print(unique_user_id)
print("Unique movies:")
print(unique_movie_id)

unique_user_id.sort()
unique_movie_id.sort()

print("\nUnique users:")
print(unique_user_id)
print("Unique movies:")
print(unique_movie_id)
#%%
# 创建从原始ID到从0开始的连续索引的映射字典
user_to_idx = {original_id: i for i, original_id in enumerate(unique_user_id)}
movie_to_idx = {original_id: i for i, original_id in enumerate(unique_movie_id)}

# 使用 .map() 方法，将映射应用到DataFrame，创建新的'user_idx'和'movie_idx'列
ratings_df['user_idx'] = ratings_df['userId'].map(user_to_idx)
ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_to_idx)

# 打印添加了新列的DataFrame的前5行，进行验证
print("\nDataFrame after remapping:")
print(ratings_df.head())

print("\n--- Data Loading and Remapping Complete! ---")
#%%
print(num_users, num_movies)
print(len(user_to_idx), len(movie_to_idx))
#%%
# 构建PyG的Edge Index
import torch

print("\n--- Step 3: Building Edge Index for PyG ---")

# 从DataFrame中提取已经映射好的用户索引和电影索引列
user_indices = ratings_df['user_idx'].values
movie_indices = ratings_df['movie_idx'].values

# 关键一步：为电影索引加上用户数量的偏移量，以区分用户和电影节点
# 电影节点的真实索引应该是 [num_users, num_users + num_movies - 1]
movie_indices_offset = movie_indices + num_users

# 创建 u -> m (用户到电影) 的边列表
# edge_index_u_to_m = torch.tensor([user_indices, movie_indices_offset], dtype=torch.long)
edge_u_to_m_numpy = np.stack([user_indices, movie_indices_offset])
edge_m_to_u_numpy = np.stack([movie_indices_offset, user_indices])
edge_index_u_to_m = torch.from_numpy(edge_u_to_m_numpy).long()

# 为了让GNN能够双向传播信息，我们需要一个无向图。
# 创建 m -> u (电影到用户) 的反向边列表
# edge_index_m_to_u = torch.tensor([movie_indices_offsetes_offset, user_indices], dtype=torch.long)
edge_index_m_to_u = torch.from_numpy(edge_m_to_u_numpy).long()

# 使用 torch.cat 将正向和反向边拼接在一起，形成最终的、完整的edge_index
total_edge_index = torch.cat([edge_index_u_to_m, edge_index_m_to_u], dim=1)

# 验证
print(f"\nNumber of nodes: {num_users + num_movies}")
print(f"Shape of final edge_index: {total_edge_index.shape}")
# 检查边的总数是否为 原始评分数 * 2
assert total_edge_index.shape[1] == 2 * len(ratings_df)
print("Edge index built successfully!")
#%%
print(len(ratings_df), ratings_df['rating'].nunique())
print(edge_index_u_to_m, edge_index_m_to_u)
print(edge_index_u_to_m.shape, edge_index_m_to_u.shape)
#%%
# 单元格 5: 数据集划分 (训练/验证/测试)
print("\n--- Step 4: Splitting the Dataset ---")

# 我们只对 u -> m 的原始交互边进行划分
num_interactions = len(ratings_df)

# 生成一个从 0 到 (num_interactions - 1) 的乱序索引
shuffled_indices = torch.randperm(num_interactions)

# 根据 8:1:1 的比例，切分索引
train_size = int(num_interactions * 0.80)
val_size = int(num_interactions * 0.10)

# 获取用于训练、验证、测试的乱序索引
train_shuffled_indices = shuffled_indices[:train_size]
val_shuffled_indices = shuffled_indices[train_size:(train_size+val_size)]
test_shuffled_indices = shuffled_indices[(train_size+val_size):]

# 根据切分好的索引，从原始的 u -> m 边中提取出对应的边
train_edge_index = edge_index_u_to_m[:, train_shuffled_indices]
val_edge_index = edge_index_u_to_m[:, val_shuffled_indices]
test_edge_index = edge_index_u_to_m[:, test_shuffled_indices]

# 【关键的验证步骤】
# 在这个单元格的最后，加入打印形状的断言，确保万无一失
assert train_edge_index.shape[0] == 2
assert val_edge_index.shape[0] == 2
assert test_edge_index.shape[0] == 2
print("All split edge_indices have a correct shape of [2, N].")

# 验证划分结果
print(f"\nTotal interactions: {num_interactions}")
print(f"Training edges: {train_edge_index.shape[1]}")
print(f"Validation edges: {val_edge_index.shape[1]}")
print(f"Test edges: {test_edge_index.shape[1]}")
assert train_edge_index.shape[1] + val_edge_index.shape[1] + test_edge_index.shape[1] == num_interactions
print("Dataset split successfully!")
#%%
# 单元格 6: 封装为PyG的Data对象并保存

from torch_geometric.data import Data

print("\n--- Step 5: Encapsulating data into PyG Data object and saving ---")
data = Data()

# 存储节点数量信息
data.num_users = num_users
data.num_movies = num_movies
data.num_nodes = num_users + num_movies

# LightGCN模型在传播时，需要用到图的完整结构（但只应包含训练数据）。
# 所以，我们将训练集的边（正向+反向）作为模型进行邻域聚合时使用的edge_index。
# 首先，我们需要构建一个包含训练集正向和反向边的 train_total_edge_index
train_edge_index_u_to_m = train_edge_index
# 我们要创建 m->u 的反向边。这其实就是交换 train_edge_index 的两行。
# 最简单、最高效的方法是直接使用张量索引来构建一个新的张量。
# train_edge_index[1] 是所有电影节点（终点）
# train_edge_index[0] 是所有用户节点（起点）
# 我们用 torch.stack 将这两个一维张量沿着新的维度（dim=0）堆叠起来，
# 形成一个新的 [2, num_edges] 的张量。
train_edge_index_m_to_u = torch.stack([train_edge_index[1], train_edge_index[0]], dim=0)
train_total_edge_index = torch.cat([train_edge_index_u_to_m, train_edge_index_m_to_u], dim=1)


# 将这个用于模型传播的边索引存入data对象
data.edge_index = train_total_edge_index

# 将划分好的、仅包含 u->m 交互的边也存进去，用于后续的训练和评估
data.train_pos_edge_index = train_edge_index
data.val_pos_edge_index = val_edge_index
data.test_pos_edge_index = test_edge_index

# 3. 打印Data对象，检查我们的“总装”成果
print("\nFinal Data object structure:")
print(data)

# 4. 将最终的“成品”打包保存到磁盘
#    这样我们未来就不需要再重复以上所有步骤了
output_path = "data.pt"
torch.save(data, output_path)

print(f"\nData object successfully assembled and saved to '{output_path}'!")

# 5. (可选但推荐) 验证加载
print("\nVerifying saved data...")
loaded_data = torch.load(output_path, weights_only=False)
print("Loaded data object structure:")
print(loaded_data)
assert str(data) == str(loaded_data) # 简单验证内容是否一致
print("Verification successful!")
#%%
