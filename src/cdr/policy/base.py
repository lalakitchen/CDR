from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path


FeatureFn = Callable[[int, int], np.ndarray]  # feature(u,i) -> 1D vector


class Policy(ABC):
    """Base interface for bandit / RL re-ranking policies."""

    @abstractmethod
    def rank(
        self,
        user_id: int,
        cand_items: np.ndarray,
        base_scores: np.ndarray,
        k: int,
        feature_fn: Optional[FeatureFn] = None,
    ) -> List[int]:
        """Return a list of item ids (length k) to show."""

    @abstractmethod
    def update(
        self,
        user_id: int,
        item_id: int,
        reward: float,
        feature_vec: Optional[np.ndarray] = None,
    ) -> None:
        """Update policy state from observed reward."""

    def state_dict(self) -> dict:
        """Optional serialization."""
        return {}

    def load_state_dict(self, d: dict) -> None:
        """Optional deserialization."""
        return

    # ---- helpers for saving/loading simple JSON state ----
    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    def load_json(self, path: Path) -> None:
        if path.exists():
            with open(path, "r") as f:
                self.load_state_dict(json.load(f))
