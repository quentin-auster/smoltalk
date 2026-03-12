"""Loss functions: CTC and contrastive (InfoNCE) for SSL."""
from .ctc import ctc_loss
from .contrastive import InfoNCELoss

__all__ = ["ctc_loss", "InfoNCELoss"]
