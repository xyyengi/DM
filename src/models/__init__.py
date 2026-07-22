"""Model construction entry points for legacy V4 and explicit V5 architectures."""

from .factory import build_model, load_model_checkpoint, resolve_architecture

__all__ = ["build_model", "load_model_checkpoint", "resolve_architecture"]
