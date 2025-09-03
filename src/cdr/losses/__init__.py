from .bpr import bpr_loss
from .hinge import hinge_loss

def get_loss(name: str):
    """
    Factory to fetch a pairwise ranking loss by name.

    Supported:
      - "bpr"   : Bayesian Personalized Ranking loss
      - "hinge" : Pairwise hinge loss (margin=1.0)

    Returns
    -------
    callable
        A function taking (s_pos, s_neg) -> scalar loss.

    Raises
    ------
    ValueError
        If the loss name is unknown.
    """
    n = name.lower()
    if n == "bpr":
        return bpr_loss
    if n == "hinge":
        # default margin=1.0; for custom margin, wrap with a lambda in your trainer
        return hinge_loss
    raise ValueError(f"Unknown loss: {name}")
