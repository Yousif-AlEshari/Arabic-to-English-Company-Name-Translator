import os, json
from dataclasses import dataclass, asdict
from collections import Counter
import torch
from vocab import Vocab
from model import Encoder, Decoder, Seq2Seq
from config import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

# --- Vocab (de)serialization ---
def vocab_to_json(vocab: Vocab) -> dict:
    return {"itos": vocab.itos}

def vocab_from_json(obj: dict) -> Vocab:
    v = Vocab(counter=Counter(), max_size=None, min_freq=1)
    v.itos = obj["itos"]
    v.stoi = {tok: i for i, tok in enumerate(v.itos)}
    return v

@dataclass
class ModelConfig:
    encoder_embedding_size: int
    decoder_embedding_size: int
    hidden_size: int
    num_layers: int
    enc_dropout: float
    dec_dropout: float

def save_artifacts(path: str, model: Seq2Seq, src_vocab: Vocab, trg_vocab: Vocab, cfg: ModelConfig):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    with open(os.path.join(path, "src_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_to_json(src_vocab), f, ensure_ascii=False)
    with open(os.path.join(path, "trg_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_to_json(trg_vocab), f, ensure_ascii=False)
    meta = asdict(cfg)
    meta.update({
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX, "unk_idx": UNK_IDX,
        "library": "pytorch", "format": "seq2seq_lstm_v1"
    })
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_artifacts(path: str, device: torch.device):
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg = ModelConfig(
        encoder_embedding_size=meta["encoder_embedding_size"],
        decoder_embedding_size=meta["decoder_embedding_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        enc_dropout=meta["enc_dropout"],
        dec_dropout=meta["dec_dropout"],
    )
    with open(os.path.join(path, "src_vocab.json"), "r", encoding="utf-8") as f:
        src_vocab = vocab_from_json(json.load(f))
    with open(os.path.join(path, "trg_vocab.json"), "r", encoding="utf-8") as f:
        trg_vocab = vocab_from_json(json.load(f))
    encoder = Encoder(len(src_vocab), cfg.encoder_embedding_size, cfg.hidden_size, cfg.num_layers, cfg.enc_dropout).to(device)
    decoder = Decoder(len(trg_vocab), cfg.decoder_embedding_size, cfg.hidden_size, len(trg_vocab), cfg.num_layers, cfg.dec_dropout).to(device)
    model = Seq2Seq(encoder, decoder, trg_vocab_size=len(trg_vocab)).to(device)
    state = torch.load(os.path.join(path, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, src_vocab, trg_vocab, cfg
