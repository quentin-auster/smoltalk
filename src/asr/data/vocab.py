"""Character-level vocabulary for CTC-based ASR."""
from __future__ import annotations

import string
from dataclasses import dataclass, field


# Special token IDs (fixed positions for easy config reference)
BLANK_ID = 0
PAD_ID = 1
UNK_ID = 2

# Printable characters used in English ASR
_ENGLISH_CHARS = list(string.ascii_lowercase + " '")  # 28 chars


@dataclass
class CharVocab:
    """Character vocabulary mapping chars ↔ IDs.

    Layout:
        0 = <blank>  (CTC blank)
        1 = <pad>    (padding)
        2 = <unk>    (unknown character)
        3+ = actual characters (space, a-z, apostrophe, ...)
    """

    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    @property
    def blank_id(self) -> int:
        return BLANK_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs (lowercased)."""
        return [self.token_to_id.get(c, UNK_ID) for c in text.lower()]

    def decode(self, ids: list[int], collapse_repeated: bool = True, remove_blank: bool = True) -> str:
        """Convert token IDs to text.

        Args:
            ids: list of token IDs (raw CTC output)
            collapse_repeated: merge consecutive identical IDs (CTC rule)
            remove_blank: remove blank tokens after collapsing
        """
        if collapse_repeated and ids:
            deduped = [ids[0]]
            for tok in ids[1:]:
                if tok != deduped[-1]:
                    deduped.append(tok)
            ids = deduped

        tokens = [self.id_to_token[i] for i in ids if i < len(self.id_to_token)]

        if remove_blank:
            tokens = [t for t in tokens if t not in ("<blank>", "<pad>")]

        return "".join(tokens).strip()


def build_char_vocab(extra_chars: list[str] | None = None) -> CharVocab:
    """Build the standard English character vocabulary.

    Returns a CharVocab with IDs:
        0 → <blank>
        1 → <pad>
        2 → <unk>
        3..N → characters from _ENGLISH_CHARS + extra_chars
    """
    specials = ["<blank>", "<pad>", "<unk>"]
    chars = _ENGLISH_CHARS + (extra_chars or [])

    # Deduplicate while preserving order
    seen: set[str] = set()
    all_tokens: list[str] = []
    for t in specials + chars:
        if t not in seen:
            all_tokens.append(t)
            seen.add(t)

    token_to_id = {t: i for i, t in enumerate(all_tokens)}
    return CharVocab(token_to_id=token_to_id, id_to_token=all_tokens)
