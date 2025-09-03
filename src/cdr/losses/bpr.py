import tensorflow as tf

@tf.function
def bpr_loss(s_pos: tf.Tensor, s_neg: tf.Tensor) -> tf.Tensor:
    """
    Bayesian Personalized Ranking (pairwise) loss.

    Args
    ----
    s_pos : (B,) float tensor
        Scores for positive (observed) items.
    s_neg : (B,) float tensor
        Scores for sampled negative items.

    Returns
    -------
    scalar tensor
        Mean pairwise loss over the batch.
    """
    # maximize s_pos - s_neg  <=>  minimize -log σ(s_pos - s_neg)
    return -tf.reduce_mean(tf.math.log_sigmoid(s_pos - s_neg))
