
from .base import Policy, FeatureFn
from .epsilon import EpsilonGreedyPolicy
from .ucb import UCB1Policy
from .linucb import LinUCBPolicy


def build_policy(name: str, **kw) -> Policy:
    n = name.lower()
    if n in ("eps", "epsilon", "e-greedy", "egreedy"):
        return EpsilonGreedyPolicy(**kw)
    if n in ("ucb", "ucb1"):
        return UCB1Policy(**kw)
    if n == "linucb":
        return LinUCBPolicy(**kw)
    raise ValueError(f"Unknown policy: {name}")
