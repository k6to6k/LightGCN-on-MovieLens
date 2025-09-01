import torch
import torch.optim as optim
from model import LightGCN
from tqdm import tqdm

print("--- Step 1: Loading Data ---")
# 加载我们之前处理好的data.pt文件
data_path = 'data.pt'
data = torch.load(data_path, weights_only=False)
print("Data loaded successfully:")
print(data)


print("\n--- Step 2: Instantiating the Model ---")
# 从data对象中获取用户和电影的数量
num_users = data['num_users']
num_movies = data['num_movies']

# 实例化我们的LightGCN模型骨架
model = LightGCN(num_users, num_movies)
print("\nModel instantiated successfully:")
print(model)


print("\n--- Step 3: Running a 'Fake' Training Loop to Test the Pipeline ---")
# 我们只关心数据流，所以只取一小批数据来测试
# 从训练集的边索引中，取前10个交互作为我们的“批次(batch)”
batch_size = 10
batch = data.train_pos_edge_index[:, :batch_size]

# 从批次中分离出用户索引和电影索引
user_indices = batch[0, :]
movie_indices = batch[1, :]

print(f"\nTesting with a batch of size: {batch_size}")
print(f"User indices shape: {user_indices.shape}")
print(f"Movie indices shape: {movie_indices.shape}")

# 将这批数据送入模型，执行前向传播
user_embeddings, movie_embeddings = model.forward(user_indices, movie_indices)

# 检查输出的形状是否正确
# 嵌入向量的形状应该是 [batch_size, embed_dim]
print("\n--- Pipeline Test Output ---")
print(f"Output user embeddings shape: {user_embeddings.shape}")
print(f"Output movie embeddings shape: {movie_embeddings.shape}")

# 验证形状是否符合预期
embed_dim = 64 # 和模型定义中保持一致
assert user_embeddings.shape == (batch_size, embed_dim)
assert movie_embeddings.shape == (batch_size, embed_dim)

print("\n🎉🎉🎉 Congratulations! The data pipeline is working correctly! 🎉🎉🎉")