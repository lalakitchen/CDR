from typing import Dict, Optional, List
import numpy as np
from numpy.linalg import inv
from .base import Policy, FeatureFn


class LinUCBPolicy(Policy):
    """
    Per-item LinUCB with features x = feature_fn(u,i):
      A_i = I + Σ x x^T
      b_i = Σ r x
      θ_i = A_i^{-1} b_i
      score = x^T θ_i + alpha * sqrt(x^T A_i^{-1} x)
    """

    def __init__(self, alpha: float = 1.0, feat_dim: Optional[int] = None):
        self.alpha = float(alpha)
        self.feat_dim = feat_dim  # can be inferred on first call
        self.A: Dict[int, np.ndarray] = {}  # item -> (d,d)
        self.b: Dict[int, np.ndarray] = {}  # item -> (d,)

    def _ensure_item(self, item_id: int, d: int) -> None:
        if item_id not in self.A:
            self.A[item_id] = np.eye(d, dtype=np.float32)
            self.b[item_id] = np.zeros(d, dtype=np.float32)

    def rank(
        self,
        user_id: int,
        cand_items: np.ndarray,
        base_scores: np.ndarray,
        k: int,
        feature_fn: Optional[FeatureFn] = None,
    ) -> List[int]:
        assert feature_fn is not None, "LinUCB needs a feature function feature(u,i)->x"
        # infer dim from first feature vector if needed
        if self.feat_dim is None:
            x0 = feature_fn(int(user_id), int(cand_items[0]))
            self.feat_dim = int(x0.shape[0])

        scores = []
        for iid in cand_items.tolist():
            self._ensure_item(int(iid), self.feat_dim)
            A = self.A[int(iid)]
            b = self.b[int(iid)]
            x = feature_fn(int(user_id), int(iid)).astype(np.float32)
            A_inv = inv(A)
            theta = A_inv @ b
            mean = float(x @ theta)
            bonus = float(np.sqrt(x @ (A_inv @ x)))
            scores.append(mean + self.alpha * bonus)

        order = np.argsort(-np.asarray(scores))[:k]
        return cand_items[order].tolist()

    def update(self, user_id: int, item_id: int, reward: float, feature_vec: Optional[np.ndarray] = None) -> None:
        if feature_vec is None:
            raise ValueError("LinUCB.update requires feature_vec")
        x = feature_vec.astype(np.float32)
        d = int(x.shape[0])
        self._ensure_item(int(item_id), d)
        self.A[int(item_id)] += np.outer(x, x)
        self.b[int(item_id)] += float(reward) * x

    def state_dict(self) -> dict:
        # Only serialize items we touched to keep it small
        out = {"alpha": self.alpha, "feat_dim": self.feat_dim, "A": {}, "b": {}}
        for k, A in self.A.items():
            out["A"][str(k)] = A.tolist()
        for k, b in self.b.items():
            out["b"][str(k)] = b.tolist()
        return out

    def load_state_dict(self, d: dict) -> None:
        self.alpha = float(d.get("alpha", self.alpha))
        self.feat_dim = d.get("feat_dim", self.feat_dim)
        self.A = {int(k): np.array(v, dtype=np.float32) for k, v in d.get("A", {}).items()}
        self.b = {int(k): np.array(v, dtype=np.float32) for k, v in d.get("b", {}).items()}
