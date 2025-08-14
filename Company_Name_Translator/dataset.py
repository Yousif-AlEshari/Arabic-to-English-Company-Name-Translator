from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from config import PAD_IDX, SOS_IDX, EOS_IDX
from vocab import Vocab

class TranslationDataset(Dataset):
    def __init__(self, split, src_tokenize, trg_tokenize, src_vocab: Vocab, trg_vocab: Vocab):
        self.data = split
        self.src_tok = src_tokenize
        self.trg_tok = trg_tokenize
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
    def __len__(self): return len(self.data)
    def _get_item(self, idx):
        rec = self.data[idx]
        return rec["translation"]["ar"], rec["translation"]["en"]
    def _encode(self, tokens, vocab):
        return [SOS_IDX] + vocab.lookup_indices(tokens) + [EOS_IDX]
    def __getitem__(self, idx):
        src_text, trg_text = self._get_item(idx)
        src_ids = self._encode(self.src_tok(src_text), self.src_vocab)
        trg_ids = self._encode(self.trg_tok(trg_text), self.trg_vocab)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    src_seqs, trg_seqs = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=False, padding_value=PAD_IDX)
    trg_padded = nn.utils.rnn.pad_sequence(trg_seqs, batch_first=False, padding_value=PAD_IDX)
    return src_padded, trg_padded
