# LightGCN-on-MovieLens

## 2.Graph Construction

Our graph is a **bipartite graph** consisting of two types of nodes: **users** and **movies**.

### 2.1 Building `edge_index`

1. Load `ratings.dat`.
2. Extract unique user IDs and movie IDs
3. Create two mappings:
    - `user_to_idx`: Maps original `UserID` to a continuous index `0, 1, ..., N-1`.
    - `movie_to_idx`: Maps original `MovieID` to a continuous index `0, 1, ..., M-1`.
4.  Iterate through `ratings.dat`:
    - For each (UserID, MovieID) pair:
        - `u_idx = user_to_idx[UserID]`
        - `m_idx = movie_to_idx[MovieID] + num_users`  // Offset movie indices!
        - Add the edge `(u_idx, m_idx)` to our edge list.
5.  Convert the edge list into a `[2, num_ratings]` tensor for `edge_index`. For GNNs, we usually treat it as undirected, so we add reverse edges `(m_idx, u_idx)` as well.

### 2.2 Handling Node Features (`data.x`)

The MovieLens dataset lacks rich initial node features. Following the LightGCN paper, we will not use handcrafted features. Instead, we will use **learnable embeddings** as the initial features.

- A `torch.nn.Embedding` layer will be created in the model, with a size of `(num_users + num_movies, embedding_dim)`.
- Each node's initial feature is its corresponding row in this embedding matrix, which will be learned during the training process.

