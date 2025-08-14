# Special tokens & indices (shared everywhere)
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# Default training hyperparams (you can change in train.py if you like)
ENC_EMB = 300
DEC_EMB = 300
HID = 512
LAYERS = 4
DROPOUT = 0.5
LR = 1e-3
BATCH = 64
EPOCHS = 10

# Artifacts folder
ARTIFACTS_DIR = "artifacts_ar_en"
