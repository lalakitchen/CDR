import tensorflow as tf
from typing import Iterable, Tuple

class NeuMF(tf.keras.Model):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        d: int,
        l2: float,
        mlp_layers: Iterable[int] = None,
        dropout: float = 0.0,
        init_std: float = 0.01,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = (2 * d, d)

        init = tf.keras.initializers.RandomNormal(stddev=init_std)
        reg_emb   = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None
        reg_dense = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

        # GMF embeddings
        self.U_gmf = tf.keras.layers.Embedding(
            n_users, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="user_emb_gmf"
        )
        self.V_gmf = tf.keras.layers.Embedding(
            n_items, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="item_emb_gmf"
        )

        # MLP embeddings
        self.U_mlp = tf.keras.layers.Embedding(
            n_users, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="user_emb_mlp"
        )
        self.V_mlp = tf.keras.layers.Embedding(
            n_items, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="item_emb_mlp"
        )

        # MLP tower
        self.mlp = tf.keras.Sequential(name="mlp_tower")
        for h in mlp_layers:
            self.mlp.add(tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg_dense))
            if dropout and dropout > 0:
                self.mlp.add(tf.keras.layers.Dropout(dropout))

        # Final fusion
        self.fuse = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=reg_dense, name="fuse")

    def score(self, u: tf.Tensor, i: tf.Tensor, training: bool = False) -> tf.Tensor:
        # ensure correct dtype for embedding lookup
        u = tf.cast(u, tf.int32)
        i = tf.cast(i, tf.int32)

        # GMF path
        ug = self.U_gmf(u)                # (B, d)
        vg = self.V_gmf(i)                # (B, d)
        gmf_vec = ug * vg                 # (B, d)

        # MLP path
        um = self.U_mlp(u)                # (B, d)
        vm = self.V_mlp(i)                # (B, d)
        mlp_in  = tf.concat([um, vm], axis=1)       # (B, 2d)
        mlp_out = self.mlp(mlp_in, training=training)  # pass training flag

        # Fuse
        z = tf.concat([gmf_vec, mlp_out], axis=1)   # (B, d + hL)
        s = tf.squeeze(self.fuse(z), axis=1)        # (B,)
        return s

    def call(self, inputs, training: bool = False):
        u, i = inputs
        return self.score(u, i, training=training)

    def get_embeddings(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # Convention: return GMF embeddings
        return self.U_gmf.embeddings, self.V_gmf.embeddings
