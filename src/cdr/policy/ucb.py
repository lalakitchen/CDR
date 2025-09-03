import math
from typing import Dict, Optional, List
import numpy as np
from .base import Policy, FeatureFn


class UCB1Policy(Policy):
    """
    Global item-level UCB1 on top of base scores.
    score = base_score + alpha * sqrt(2 ln T / (n_i + 1e-9))
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.counts: Dict[int, int] = {}    # pulls per item
        self.rewards: Dict[int, float] = {} # total reward per item
        self.T: int = 0                     # total pulls

    def _bonus(self, i: int) -> float:
        n_i = self.counts.get(i, 0)
        if n_i == 0:
            # encourage unseen items
            return 1e3
        return math.sqrt(2.0 * math.log(max(self.T, 1)) / n_i)

    def rank(
        self,
        user_id: int,
        cand_items: np.ndarray,
        base_scores: np.ndarray,
        k: int,
        feature_fn: Optional[FeatureFn] = None,
    ) -> List[int]:
        # normalize base scores to [0,1] for a cleaner mix
        if len(base_scores) > 1:
            mn, mx = float(base_scores.min()), float(base_scores.max())
            base = (base_scores - mn) / (mx - mn + 1e-12)
        else:
            base = base_scores

        ucb = np.array([base[t] + self.alpha * self._bonus(int(cand_items[t])) for t in range(len(cand_items))])
        order = np.argsort(-ucb)[:k]
        chosen = cand_items[order].tolist()
        # count pulls
        for iid in chosen:
            self.counts[iid] = self.counts.get(iid, 0) + 1
            self.T += 1
        return chosen

    def update(self, user_id: int, item_id: int, reward: float, feature_vec=None) -> None:
        self.rewards[item_id] = self.rewards.get(item_id, 0.0) + float(reward)

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "counts": self.counts, "rewards": self.rewards, "T": self.T}

    def load_state_dict(self, d: dict) -> None:
        self.alpha = float(d.get("alpha", self.alpha))
        self.counts = {int(k): int(v) for k, v in d.get("counts", {}).items()}
        self.rewards = {int(k): float(v) for k, v in d.get("rewards", {}).items()}
        self.T = int(d.get("T", self.T))
