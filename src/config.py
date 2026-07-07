import torch

class Config:
    VOCAB_SIZE = 5000
    EMBED_DIM = 256
    BATCH_SIZE = 64
    SEQ_LEN = 35
    EPOCAS = 20
    DATA_LIMIT = 600000
    NUM_CAPAS = 3
    LR = 0.0005
    SAVE_PATH = "chamber_v4_master.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
