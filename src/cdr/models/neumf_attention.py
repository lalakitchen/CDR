import tensorflow as tf
from typing import Iterable, Tuple, Optional

class NeuMFAttention(tf.keras.Model):
    """
    NeuMF with QKV cross-attention between user and item embeddings.

    Paths:
      - GMF: (u_gmf * i_gmf)
      - MLP: MLP([u_mlp, i_mlp])
      - QKV: two cross-attention blocks (U→V and V→U) over tokens [gmf, mlp],
             pooled to (B, d), combined with GMF via elementwise product.

    Final fusion: concat([GMF, QKV_GMF, MLP]) -> Dense(1)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        d: int,
        l2: float,
        mlp_layers: Iterable[int] = None,
        dropout: float = 0.0,
        init_std: float = 0.01,
        # QKV settings
        num_heads: int = 2,
        attn_dropout: float = 0.0,
        attn_key_dim: Optional[int] = None,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = (2 * d, d)

        init = tf.keras.initializers.RandomNormal(stddev=init_std)
        reg_emb   = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None
        reg_dense = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.d = int(d)

        # ---- GMF embeddings ----
        self.U_gmf = tf.keras.layers.Embedding(
            n_users, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="user_emb_gmf"
        )
        self.V_gmf = tf.keras.layers.Embedding(
            n_items, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="item_emb_gmf"
        )

        # ---- MLP embeddings ----
        self.U_mlp = tf.keras.layers.Embedding(
            n_users, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="user_emb_mlp"
        )
        self.V_mlp = tf.keras.layers.Embedding(
            n_items, d, embeddings_initializer=init, embeddings_regularizer=reg_emb, name="item_emb_mlp"
        )

        # ---- MLP tower ----
        self.mlp = tf.keras.Sequential(name="mlp_tower")
        for h in mlp_layers:
            self.mlp.add(tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg_dense))
            if dropout and dropout > 0:
                self.mlp.add(tf.keras.layers.Dropout(dropout))

        # ---- QKV cross-attention ----
        # Per-head key dim: pick a safe default if d isn't divisible by heads.
        if attn_key_dim is None:
            attn_key_dim = d // num_heads if d % num_heads == 0 else max(1, d // num_heads)

        # Force MHA output to have last dim d for stability across TF versions
        self.mha_u2v = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=attn_key_dim, dropout=attn_dropout,
            output_shape=d, name="u2v_attn"
        )
        self.mha_v2u = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=attn_key_dim, dropout=attn_dropout,
            output_shape=d, name="v2u_attn"
        )
        # Small linear “post” to stabilize and keep shape
        self.post_u = tf.keras.layers.Dense(d, activation="linear", kernel_regularizer=reg_dense, name="post_u")
        self.post_v = tf.keras.layers.Dense(d, activation="linear", kernel_regularizer=reg_dense, name="post_v")

        # ---- Final fusion ----
        # concat([GMF d, QKV d, last_mlp]) -> 1
        self.fuse = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=reg_dense, name="fuse")

    # -------- QKV block --------
    def _qkv_block(self, ug, um, vg, vm, training: bool) -> tf.Tensor:
        """
        U_seq = [u_gmf, u_mlp], V_seq = [v_gmf, v_mlp]  -> (B, 2, d)
        Returns GMF-style interaction on attended reps: (B, d)
        """
        U_seq = tf.stack([ug, um], axis=1)  # (B, 2, d)
        V_seq = tf.stack([vg, vm], axis=1)  # (B, 2, d)

        # U attends to V; V attends to U
        att_u = self.mha_u2v(query=U_seq, value=V_seq, key=V_seq, training=training)  # (B,2,d)
        att_v = self.mha_v2u(query=V_seq, value=U_seq, key=U_seq, training=training)  # (B,2,d)

        # Pool across tokens (mean);  can try first-token as an ablation
        u_att = tf.reduce_mean(att_u, axis=1)  # (B,d)
        v_att = tf.reduce_mean(att_v, axis=1)  # (B,d)

        u_att = self.post_u(u_att)
        v_att = self.post_v(v_att)

        return u_att * v_att  # (B,d)

    # -------- scoring --------
    def score(self, u: tf.Tensor, i: tf.Tensor, training: bool = False) -> tf.Tensor:
        u = tf.cast(u, tf.int32)
        i = tf.cast(i, tf.int32)

        ug = self.U_gmf(u)   # (B,d)
        vg = self.V_gmf(i)   # (B,d)
        um = self.U_mlp(u)   # (B,d)
        vm = self.V_mlp(i)   # (B,d)

        gmf_vec = ug * vg                                   # (B,d)
        mlp_in  = tf.concat([um, vm], axis=1)               # (B,2d)
        mlp_out = self.mlp(mlp_in, training=training)       # (B,hL)

        qkv_gmf = self._qkv_block(ug, um, vg, vm, training) # (B,d)
        z = tf.concat([gmf_vec, qkv_gmf, mlp_out], axis=1)  # (B, d + d + hL)

        s = tf.squeeze(self.fuse(z), axis=1)                # (B,)
        return s

    def call(self, inputs, training: bool = False):
        u, i = inputs
        return self.score(u, i, training=training)

    def get_embeddings(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Expose GMF embeddings for exporters/feature functions."""
        return self.U_gmf.embeddings, self.V_gmf.embeddings
