import tensorflow as tf
import numpy as np
from typing import Optional


class LightGCN(tf.keras.Model):
    """
    LightGCN (He et al., SIGIR'20) for implicit recommendation.
    - Trainable user/item embeddings
    - Propagation over normalized user-item graph (no nonlinearities/weights)
    - Final embedding is the average of layer-wise embeddings (0..K)

    Args
    ----
    n_users, n_items : ints
    d                : embedding dim
    l2               : L2 reg on embeddings
    norm_adj         : tf.sparse.SparseTensor (shape [n_users+n_items, n_users+n_items])
                       normalized bipartite adjacency built from TRAIN interactions
    n_layers         : propagation layers (default 3)

    Methods
    -------
    score(u, i): (B,) dot product between propagated user/item embeddings
    """

    def __init__(self, n_users: int, n_items: int, d: int, l2: float,
                 norm_adj: tf.sparse.SparseTensor, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = int(n_layers)
        reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

        self.U = tf.keras.layers.Embedding(
            n_users, d,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            embeddings_regularizer=reg,
            name="user_emb",
        )
        self.V = tf.keras.layers.Embedding(
            n_items, d,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            embeddings_regularizer=reg,
            name="item_emb",
        )
        # store normalized adjacency (constant during training)
        self.norm_adj = norm_adj

    def _propagate(self) -> tf.Tensor:
        """
        Returns stacked embeddings [E0, E1, ..., EK] of shape (K+1, N, d),
        where N = n_users + n_items, and final embedding is their mean.
        """
        E0_u = self.U(tf.range(self.n_users, dtype=tf.int32))           # (U, d)
        E0_i = self.V(tf.range(self.n_items, dtype=tf.int32))           # (I, d)
        E = tf.concat([E0_u, E0_i], axis=0)                             # (N, d)
        outs = [E]

        for _ in range(self.n_layers):
            E = tf.sparse.sparse_dense_matmul(self.norm_adj, E)         # (N, d)
            outs.append(E)

        return tf.stack(outs, axis=0)                                   # (K+1, N, d)

    def _final_embeddings(self) -> tf.Tensor:
        # mean of layer-wise embeddings (LightGCN)
        Es = self._propagate()                                          # (K+1, N, d)
        return tf.reduce_mean(Es, axis=0)                               # (N, d)

    def score(self, u: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """
        u, i: int tensors (B,)
        """
        E = self._final_embeddings()                                    # (N, d)
        Ue = tf.gather(E, u)                                            # (B, d)  (users index from 0..U-1)
        Ie = tf.gather(E, self.n_users + i)                             # (B, d)  (items offset by n_users)
        return tf.reduce_sum(Ue * Ie, axis=1)

    def call(self, inputs):
        u, i = inputs
        return self.score(u, i)
