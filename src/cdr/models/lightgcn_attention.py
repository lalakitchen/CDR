# lightgcn_attention.py
import tensorflow as tf
import numpy as np
from typing import Optional


class LightGCNAttention(tf.keras.Model):
    """
    LightGCN with layer-wise QKV self-attention.

    Propagation: same as LightGCN on a fixed normalized user–item graph.
    Aggregation: instead of mean over [E0..EK], use attention over layers:
      - Query from E0 (per node)
      - Keys and Values from each layer embedding
      - Multi-head optional, softmax over layers per node

    Args
    ----
    n_users, n_items : ints
    d                : base embedding dim
    l2               : L2 regularization on base embeddings
    norm_adj         : tf.sparse.SparseTensor of shape [U+I, U+I]
    n_layers         : number of propagation layers (K)
    num_heads        : attention heads
    head_dim         : per-head key/query dim (defaults to d // num_heads)
    value_dim        : per-head value dim (defaults to d // num_heads)
    temperature      : softmax temperature (>0); lower is sharper
    attn_dropout     : dropout rate on attention weights during training
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        d: int,
        l2: float,
        norm_adj: tf.sparse.SparseTensor,
        n_layers: int = 3,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        temperature: float = 1.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_layers = int(n_layers)

        self.num_heads = int(num_heads)
        self.d = int(d)
        self.d_k = int(head_dim) if head_dim is not None else max(1, d // self.num_heads)
        self.d_v = int(value_dim) if value_dim is not None else max(1, d // self.num_heads)

        self.temperature = float(temperature)
        self.attn_dropout_rate = float(attn_dropout)

        reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

        # Base trainable embeddings
        self.U = tf.keras.layers.Embedding(
            self.n_users, d,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            embeddings_regularizer=reg,
            name="user_emb",
        )
        self.V = tf.keras.layers.Embedding(
            self.n_items, d,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            embeddings_regularizer=reg,
            name="item_emb",
        )

        # Fixed normalized adjacency from train interactions
        self.norm_adj = norm_adj  # tf.sparse.SparseTensor

        # QKV projections (per head)
        glorot = tf.keras.initializers.GlorotUniform()
        self.Wq = self.add_weight(
            name="Wq", shape=(self.d, self.num_heads, self.d_k),
            initializer=glorot, trainable=True,
        )
        self.Wk = self.add_weight(
            name="Wk", shape=(self.d, self.num_heads, self.d_k),
            initializer=glorot, trainable=True,
        )
        self.Wv = self.add_weight(
            name="Wv", shape=(self.d, self.num_heads, self.d_v),
            initializer=glorot, trainable=True,
        )
        # Output projection to bring concat heads back to d
        self.Wo = self.add_weight(
            name="Wo", shape=(self.num_heads * self.d_v, self.d),
            initializer=glorot, trainable=True,
        )

        self._attn_dropout = tf.keras.layers.Dropout(rate=self.attn_dropout_rate)
        self._cached_final: Optional[tf.Tensor] = None  # (N, d)

    # ---------------- propagation ----------------
    def _propagate(self) -> tf.Tensor:
        """
        Returns stacked embeddings [E0, E1, ..., EK] of shape (K+1, N, d),
        where N = n_users + n_items.
        """
        E0_u = self.U(tf.range(self.n_users, dtype=tf.int32))  # (U, d)
        E0_i = self.V(tf.range(self.n_items, dtype=tf.int32))  # (I, d)
        E = tf.concat([E0_u, E0_i], axis=0)                    # (N, d)
        outs = [E]
        for _ in range(self.n_layers):
            E = tf.sparse.sparse_dense_matmul(self.norm_adj, E)  # (N, d)
            outs.append(E)
        return tf.stack(outs, axis=0)  # (K+1, N, d)

    # ---------------- attention over layers ----------------
    def _project_q(self, E0: tf.Tensor) -> tf.Tensor:
        """
        E0: (N, d)
        returns Q: (N, H, d_k)
        """
        # einsum 'nd,dhk->nhk'
        return tf.einsum("nd,dhk->nhk", E0, self.Wq)

    def _project_kv(self, Es: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Es: (L, N, d) where L = K+1
        returns K: (L, N, H, d_k), V: (L, N, H, d_v)
        """
        K = tf.einsum("lnd,dhk->lnhk", Es, self.Wk)
        V = tf.einsum("lnd,dhv->lnhv", Es, self.Wv)
        return K, V

    def _aggregate_qkv(self, Es: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Es: (L, N, d)
        Q from E0 per node, Keys/Values from each layer embedding.
        Softmax over layers for each node and head.
        Returns (N, d)
        """
        E0 = Es[0]                               # (N, d)
        Q = self._project_q(E0)                  # (N, H, d_k)
        K, V = self._project_kv(Es)              # K:(L,N,H,d_k)  V:(L,N,H,d_v)

        # scores[l, n, h] = <K[l,n,h,:], Q[n,h,:]> / sqrt(d_k)
        scale = tf.cast(self.d_k, tf.float32) ** 0.5
        scores = tf.einsum("lnhd,nhd->lnh", K, Q) / scale          # (L, N, H)

        # softmax over layers for each node and head
        alphas = tf.nn.softmax(scores / self.temperature, axis=0)  # (L, N, H)
        if training and self.attn_dropout_rate > 0.0:
            # apply dropout independently on attention weights
            alphas = self._attn_dropout(alphas, training=True)

        # weighted sum over layers: sum_l alpha[l,n,h] * V[l,n,h,:]
        heads = tf.einsum("lnh,lnhd->nhd", alphas, V)              # (N, H, d_v)

        # concat heads and project to d
        N = tf.shape(heads)[0]
        heads_flat = tf.reshape(heads, [N, self.num_heads * self.d_v])  # (N, H*d_v)
        E_out = tf.matmul(heads_flat, self.Wo)                           # (N, d)
        return E_out

    # ---------------- public APIs ----------------
    def recompute_cache(self, training: bool = False) -> None:
        """
        Precompute and cache final embeddings. Call this per epoch
        to speed up scoring during training and evaluation.
        """
        Es = self._propagate()                         # (L, N, d)
        self._cached_final = self._aggregate_qkv(Es, training=training)  # (N, d)

    def _final_embeddings(self) -> tf.Tensor:
        """
        Final (N, d) embeddings, using cache when available.
        Kept for compatibility with r exporter/feature_fn.
        """
        if self._cached_final is not None:
            return self._cached_final
        Es = self._propagate()
        return self._aggregate_qkv(Es, training=False)

    def get_embeddings(self):
        """
        Convenience split into user and item blocks.
        """
        E = self._final_embeddings()
        Ue = E[: self.n_users]
        Ie = E[self.n_users : self.n_users + self.n_items]
        return Ue, Ie

    # ---------------- scoring ----------------
    def score(self, u: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """
        u, i: int tensors (B,)
        returns dot product between user and item final embeddings
        """
        E = self._final_embeddings()                # (N, d)
        Ue = tf.gather(E, u)                        # (B, d)
        Ie = tf.gather(E, self.n_users + i)         # (B, d)
        return tf.reduce_sum(Ue * Ie, axis=1)

    def call(self, inputs):
        u, i = inputs
        return self.score(u, i)
