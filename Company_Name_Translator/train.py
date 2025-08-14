import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    ARTIFACTS_DIR, ENC_EMB, DEC_EMB, HID, LAYERS, DROPOUT, LR, BATCH, EPOCHS,
    PAD_IDX
)
from tokenizers import tokenize_ar, tokenize_en
from vocab import build_vocab
from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq
from routing import translate_smart
from data_loading import load_ar_en_splits
from io_artifacts import ModelConfig, save_artifacts, load_artifacts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(os.path.join(ARTIFACTS_DIR, "model.pt")):
        print(f"Loading saved model from {ARTIFACTS_DIR} …")
        model, src_vocab, trg_vocab, cfg = load_artifacts(ARTIFACTS_DIR, device)
    else:
        print("No saved model found. Training once (online if HF is available)…")
        splits = load_ar_en_splits()
        print("Building vocabularies …")
        src_vocab = build_vocab(splits["train"], tokenize_ar, max_size=2048, min_freq=1)
        trg_vocab = build_vocab(splits["train"], tokenize_en, max_size=2048, min_freq=1)
        print(f"Vocab sizes — AR: {len(src_vocab)}  EN: {len(trg_vocab)}")

        train_ds = TranslationDataset(splits["train"], tokenize_ar, tokenize_en, src_vocab, trg_vocab)
        valid_ds = TranslationDataset(splits["validation"], tokenize_ar, tokenize_en, src_vocab, trg_vocab)
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

        cfg = ModelConfig(
            encoder_embedding_size=ENC_EMB,
            decoder_embedding_size=DEC_EMB,
            hidden_size=HID,
            num_layers=LAYERS,
            enc_dropout=DROPOUT,
            dec_dropout=DROPOUT,
        )

        encoder = Encoder(len(src_vocab), cfg.encoder_embedding_size, cfg.hidden_size, cfg.num_layers, cfg.enc_dropout).to(device)
        decoder = Decoder(len(trg_vocab), cfg.decoder_embedding_size, cfg.hidden_size, len(trg_vocab), cfg.num_layers, cfg.dec_dropout).to(device)
        model = Seq2Seq(encoder, decoder, trg_vocab_size=len(trg_vocab)).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        for epoch in range(EPOCHS):
            model.train()
            total = 0.0
            for src, trg in train_loader:
                src, trg = src.to(device), trg.to(device)
                outputs = model(src, trg)
                logits = outputs[1:].reshape(-1, outputs.shape[2])
                gold   = trg[1:].reshape(-1)
                optimizer.zero_grad()
                loss = criterion(logits, gold)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()

            # validation
            model.eval()
            with torch.no_grad():
                val_loss, count = 0.0, 0
                for src, trg in valid_loader:
                    src, trg = src.to(device), trg.to(device)
                    outputs = model(src, trg)
                    logits = outputs[1:].reshape(-1, outputs.shape[2])
                    gold   = trg[1:].reshape(-1)
                    val_loss += criterion(logits, gold).item()
                    count += 1
            print(f"Epoch {epoch+1} | Val Loss: {val_loss / max(1, count):.4f}")

        print(f"Saving artifacts to {ARTIFACTS_DIR} …")
        save_artifacts(ARTIFACTS_DIR, model, src_vocab, trg_vocab, cfg)

    # Offline inference demo
    examples = [
        "شركة القاهرة القابضة ش.م.ع",
        "مجموعة بن لادن السعودية",
        "مصرف الرافدين الإسلامي",
        "رجل يقود دراجة على الطريق بجانب النهر."
    ]
    print("\n=== Demo ===")
    for ex in examples:
        print("AR:", ex)
        print("EN:", translate_smart(model, ex, src_vocab, trg_vocab, device))

    print("\nDone.")

    # Optional Excel evaluation (offline if file exists)
    try:
        data = pd.read_excel("manual_input.xlsx").head(100)
        for arabic, english in zip(data["Original_Arabic_Name"], data["Translated_Output"]):
            print(f"\nAR: {arabic}\nEN: {translate_smart(model, arabic, src_vocab, trg_vocab, device)}\nactual translation: {english}\n")
    except Exception as e:
        print(f"Skipped manual_input.xlsx evaluation ({e}).")

if __name__ == "__main__":
    main()
