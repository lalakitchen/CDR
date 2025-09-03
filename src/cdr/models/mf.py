import tensorflow as tf
from typing import Tuple


class MF(tf.keras.Model):
    """
    Matrix Factorization with dot-product scoring.

    Args
    ----
    n_users : int
        Number of distinct users (max uid + 1).
    n_items : int
        Number of distinct items (max iid + 1).
    d : int
        Embedding dimension.
    l2 : float
        L2 weight for embedding regularization.
    use_bias : bool, optional
        If True, adds user/item bias terms and a global bias. Default False.
    init_std : float, optional
        Stddev for RandomNormal initializer. Default 0.01.

    Methods
    -------
    score(u, i) -> tf.Tensor
        Returns a (B,) tensor of scores for batches of user/item ids.
    get_embeddings() -> Tuple[tf.Tensor, tf.Tensor]
        Returns (user_emb_matrix, item_emb_matrix).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        d: int,
        l2: float,
        use_bias: bool = False,
        init_std: float = 0.01,
    ):
        super().__init__()
        init = tf.keras.initializers.RandomNormal(stddev=init_std)
        reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

        self.U = tf.keras.layers.Embedding(
            input_dim=n_users,
            output_dim=d,
            embeddings_initializer=init,
            embeddings_regularizer=reg,
            name="user_emb",
        )
        self.V = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=d,
            embeddings_initializer=init,
            embeddings_regularizer=reg,
            name="item_emb",
        )

        self.use_bias = use_bias
        if use_bias:
            self.ub = tf.keras.layers.Embedding(
                input_dim=n_users, output_dim=1,
                embeddings_initializer=tf.keras.initializers.Zeros(),
                name="user_bias",
            )
            self.vb = tf.keras.layers.Embedding(
                input_dim=n_items, output_dim=1,
                embeddings_initializer=tf.keras.initializers.Zeros(),
                name="item_bias",
            )
            # Global bias as a trainable scalar
            self.gb = tf.Variable(0.0, trainable=True, name="global_bias", dtype=tf.float32)

    def score(self, u: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """
        Compute scores for (user, item) pairs.

        Parameters
        ----------
        u : tf.int32/int64 tensor of shape (B,)
            User ids.
        i : tf.int32/int64 tensor of shape (B,)
            Item ids.

        Returns
        -------
        tf.Tensor (B,)
            Dot-product scores (+ optional biases).
        """
        uvec = self.U(u)          # (B, d)
        ivec = self.V(i)          # (B, d)
        s = tf.reduce_sum(uvec * ivec, axis=1)  # (B,)

        if self.use_bias:
            s = s + tf.squeeze(self.ub(u), axis=1) + tf.squeeze(self.vb(i), axis=1) + self.gb
        return s

    # Optional: make Keras happy if someone calls model(inputs)
    def call(self, inputs):
        u, i = inputs
        return self.score(u, i)

    def get_embeddings(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Return the full user/item embedding matrices (not numpy)."""
        return self.U.embeddings, self.V.embeddings
