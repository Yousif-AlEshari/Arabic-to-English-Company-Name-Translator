import random
import torch
import torch.nn as nn
from config import PAD_IDX, SOS_IDX, EOS_IDX
from tokenizers import tokenize_ar
from vocab import Vocab

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)  # (1, N)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs).squeeze(0)  # (N, vocab_size)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_vocab_size = trg_vocab_size
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        outputs = torch.zeros(target_len, batch_size, self.trg_vocab_size, device=source.device)
        hidden, cell = self.encoder(source)
        x = target[0]  # <sos>
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs

def greedy_translate(model: Seq2Seq, sentence_ar: str, src_vocab: Vocab, trg_vocab: Vocab, device, max_len=50):
    model.eval()
    with torch.no_grad():
        src_tokens = [SOS_IDX] + src_vocab.lookup_indices(tokenize_ar(sentence_ar)) + [EOS_IDX]
        src = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(1)  # (L, 1)
        hidden, cell = model.encoder(src)
        x = torch.tensor([SOS_IDX], dtype=torch.long, device=device)
        out_tokens = []
        for _ in range(max_len):
            logits, hidden, cell = model.decoder(x, hidden, cell)
            x = logits.argmax(1)
            token_id = x.item()
            if token_id == EOS_IDX:
                break
            if token_id not in (PAD_IDX, SOS_IDX):
                out_tokens.append(token_id)
        return " ".join(trg_vocab.lookup_tokens(out_tokens))
