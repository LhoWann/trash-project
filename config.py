import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
DATA_DIR = 'dataset'
CHECKPOINT_DIR = 'checkpoints'
MODEL_NAME = 'convnextv2_tiny'

IMG_SIZE = 224
BATCH_SIZE = 16
ACCUM_STEPS = 2
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 12
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.05