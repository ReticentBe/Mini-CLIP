"""Centralized hyperparameters and path configuration"""

import torch

DATA_PATH = './data'
CHECKPOINT_DIR = './checkpoints'            
IMAGE_DIR = './data/flickr8k/Images'
CAPTIONS_PATH = './data/flickr8k/captions.txt'
SHARDS_DIR = "data/wds"
VOCAB_PATH = './data/vocab.json'
BATCH_SIZE = 64
EPOCHS = 60
TRAIN_SAMPLES = 5000                            # Estimated training samples per epoch
VAL_SAMPLES = 500                               # Estimated validation samples per epoch
LEARNING_RATE = 3e-4
OPTIMIZER = 'adamw'
WEIGHT_DECAY = 1e-4                             # Weight decay for AdamW regularization
PROJECTION_DIM = 512                            # Shared embedding dimension for image/text
IMAGE_BACKBONE = "resnet18"
IMAGE_SIZE = 224
TEXT_MAX_LEN = 77
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
