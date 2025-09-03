# cdr/models/__init__.py

from .mf import MF
from .neumf import NeuMF
from .itemknn import ItemKNN
from .lightgcn import LightGCN

# Optional attention variants (QKV-only)
try:
    from .neumf_attention import NeuMFAttention
except Exception:
    NeuMFAttention = None

try:
    from .lightgcn_attention import LightGCNAttention
except Exception:
    LightGCNAttention = None


__all__ = [
    "MF",
    "NeuMF",
    "NeuMFAttention",
    "ItemKNN",
    "LightGCN",
    "LightGCNAttention",
    "build_model",
]


def build_model(name: str, **kwargs):
    """
    Factory for model construction.

    Supported names (case/sep-insensitive):
      - "mf"
      - "neumf"
      - "neumf_attention", "neumf_qkv"
      - "itemknn"
      - "lightgcn"
      - "lightgcn_attention", "lightgcn_qkv", "lightgcn_attn", "lgnn_attn"
    """
    n = name.lower().replace("-", "_")

    if n == "mf":
        return MF(**kwargs)

    if n == "neumf":
        return NeuMF(**kwargs)

    if n in ("neumf_attention", "neumf_qkv"):
        if NeuMFAttention is None:
            raise ImportError("NeuMFAttention not available. Ensure neumf_attention.py is present and imports cleanly.")
        return NeuMFAttention(**kwargs)

    if n == "itemknn":
        return ItemKNN(**kwargs)

    if n == "lightgcn":
        return LightGCN(**kwargs)

    if n in ("lightgcn_attention", "lightgcn_qkv", "lightgcn_attn", "lgnn_attn"):
        if LightGCNAttention is None:
            raise ImportError("LightGCNAttention not available. Ensure lightgcn_attention.py is present and imports cleanly.")
        return LightGCNAttention(**kwargs)

    raise ValueError(f"Unknown model: {name}")
