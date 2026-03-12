"""CTC decoding utilities."""
from .greedy import greedy_ctc_decode, greedy_ctc_decode_batch

__all__ = ["greedy_ctc_decode", "greedy_ctc_decode_batch"]
