import torch

DATA_PATH = './data'
CHECKPOINT_DIR = './checkpoints'
# 数据集相关路径
IMAGE_DIR = './data/flickr8k/Images'
CAPTIONS_PATH = './data/flickr8k/captions.txt'
WEBDATASET_DIR = "data/wds"
VOCAB_PATH = './data/vocab.json'
BATCH_SIZE = 256
NUM_WORKERS = 0
EPOCHS = 10
TRAIN_SAMPLES = 36000  # 训练集样本数估算
VAL_SAMPLES = 4000     # 验证集样本数估算
LEARNING_RATE = 1e-4
OPTIMIZER = 'adamw'
WEIGHT_DECAY = 1e-4   # 正则化强度
PROJECTION_DIM = 512   # embedding dim
TEMPERATURE = 0.07   
IMAGE_BACKBONE = "resnet18"
IMAGE_SIZE = 224
TEXT_MAX_LEN = 77
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TEXT_SIZE = 0.2
