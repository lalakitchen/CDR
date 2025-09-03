import numpy as np
from typing import Optional, List
from .base import Policy, FeatureFn


class EpsilonGreedyPolicy(Policy):
    """
    ε-greedy re-ranker over model candidates.
    With prob ε, swap some top items with random picks from the remaining pool.
    """

    def __init__(self, epsilon: float = 0.1, random_state: Optional[int] = None):
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(random_state)

    def rank(
        self,
        user_id: int,
        cand_items: np.ndarray,
        base_scores: np.ndarray,
        k: int,
        feature_fn: Optional[FeatureFn] = None,
    ) -> List[int]:
        order = np.argsort(-base_scores)  # greedy order
        top = order[:k].tolist()

        if self.epsilon <= 0.0 or len(cand_items) <= k:
            return cand_items[top].tolist()

        # exploration: replace ~ε*k items with random others
        n_swap = max(1, int(round(self.epsilon * k)))
        remaining = order[k:].tolist()
        if not remaining:
            return cand_items[top].tolist()

        swap_idx = self.rng.choice(k, size=n_swap, replace=False)
        rand_picks = self.rng.choice(remaining, size=n_swap, replace=False)
        for s, r in zip(swap_idx, rand_picks):
            top[s] = r
        return cand_items[top].tolist()

    def update(self, user_id: int, item_id: int, reward: float, feature_vec=None) -> None:
        # Stateless for now. You can aggregate CTR if you like.
        return

    def state_dict(self) -> dict:
        return {"epsilon": self.epsilon}

    def load_state_dict(self, d: dict) -> None:
        self.epsilon = float(d.get("epsilon", self.epsilon))
