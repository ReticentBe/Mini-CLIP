# Mini-CLIP: A Minimal Image-Text Retrieval System [WIP]

This repository contains a minimal, from-scratch implementation of a CLIP-like Vision-Language model. It serves as a foundational engineering project stepping toward Vision-Language-Action (VLA) models.



Unlike standard toy projects, this repository focuses on scalable engineering practices, including Distributed Data Parallel (DDP) training and streaming data pipeline. 



## Key Features (Current Progress)

- **Architecture**: Pluggable Vision Backbone (Custom ViT from scratch / ResNet) + Transformer Encoder (Text) + Projection Heads

- **Objective**: InfoNCE Loss (Contrastive Learning) for cross-modal alignment

- **Scalable Data Pipeline**: Refactored from Map-style datasets to Iterable-style `WebDataset`, enabling infinite streaming and efficient I/O for large-scale training.

- **Distributed Training**: Native PyTorch `DistributedDataParallel` (DDP) support with custom differentiable `all_gather` operations



## Quick Start

### 1. Environment Setup

```bash
pip install -r requirement.txt
```

### 2. Data Preparation

We use the Flickr8k dataset for initial experiments

1. Download the dataset and place images in `data/flickr8k/Images` and `captions.txt` in `data/flickr8k`

2. Pack the dataset into WebDataset `.tar` shards
   
   ```bash
   python src/pack_wds.py
   ```

### 3. Training (Multi-GPU DDP)
   
   ```bash
   torchrun --nproc_per_node=2 src/train.py
   ```

## Roadmap

- [x] Core CLIP architecture and InfoNCE loss
- [x] Multi-GPU DDP training pipeline
- [x] WebDataset integration for streaming data
- [x] Basic Inference Script for qualitative testing

- [ ] Partial Fine-tuning (Freezing backbone, tuning projection layers)
- [ ] Systematic Zero-shot Evaluation (Recall@K metrics)
- [ ] Exporting pre-trained weights for downstream VLA policies


