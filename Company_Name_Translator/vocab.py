from collections import Counter
from typing import List
from config import SPECIALS, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

class Vocab:
    def __init__(self, counter: Counter, max_size: int = 10000, min_freq: int = 2):
        tokens_freq = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
        tokens_freq.sort(key=lambda x: (-x[1], x[0]))
        tokens = [tok for tok, _ in tokens_freq]
        if max_size is not None:
            tokens = tokens[: max(0, max_size - len(SPECIALS))]
        self.itos = list(SPECIALS) + tokens
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]
    def lookup_tokens(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]

def build_vocab(split, tokenizer, max_size=2048, min_freq=2) -> "Vocab":
    counter = Counter()
    for rec in split:
        # identify direction by function object
        text = rec["translation"]["ar"] if tokenizer.__name__ == "tokenize_ar" else rec["translation"]["en"]
        counter.update(tokenizer(text))
    return Vocab(counter, max_size=max_size, min_freq=min_freq)
