import tensorflow as tf

@tf.function
def hinge_loss(s_pos: tf.Tensor, s_neg: tf.Tensor, margin: float = 1.0) -> tf.Tensor:
    """
    Pairwise hinge ranking loss.

    Args
    ----
    s_pos : (B,) float tensor
        Scores for positive (observed) items.
    s_neg : (B,) float tensor
        Scores for sampled negative items.
    margin : float
        Desired margin between positive and negative scores.

    Returns
    -------
    scalar tensor
        Mean hinge loss over the batch.
    """
    # want: s_pos >= s_neg + margin  -> loss = max(0, margin - (s_pos - s_neg))
    return tf.reduce_mean(tf.nn.relu(margin - (s_pos - s_neg)))
