"""
DDP training entry point

Initialized distributed process group, builds vocabulary (rank 0 only),
constructs WebDataset loaders, freezes ResNet backbone and runs the training
loop with CosineAnnealingLR and best-model checkpointing
"""

import os
import glob
import torch
import torch.optim as optim
from tqdm import tqdm 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
 
import config
import vocab
from dataset import build_dataloader
from model import CLIP
from loss import CLIPLoss
from engine import train_one_epoch, evaluate

def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')

    if global_rank == 0:
        if not os.path.exists(config.VOCAB_PATH):
            all_captions = []
            if os.path.exists(config.CAPTIONS_PATH):
                with open(config.CAPTIONS_PATH, "r", encoding='utf-8') as f:
                    next(f)
                    for line in f:
                        parts = line.strip().split(',', 1)
                        if len(parts) == 2:
                            all_captions.append(parts[1])
                token_to_id, _ = vocab.build_vocabulary(all_captions)
                vocab.save_vocab(token_to_id, config.VOCAB_PATH)
    dist.barrier()
    # Build vocabulary on rank 0, then broadcast via barrier
    
    token_to_id = vocab.load_vocab(config.VOCAB_PATH)

    shards_pattern = os.path.join(config.SHARDS_DIR, "*.tar")
    all_tars = sorted(glob.glob(shards_pattern))

    if len(all_tars) == 0:
        raise ValueError(f"No tar files found in {config.SHARDS_DIR}!")
    
    train_urls = all_tars[:-3]
    val_urls = all_tars[-3:]  
    # Split shards: first N-3 for training, last 3 for validation

    train_steps = config.TRAIN_SAMPLES // config.BATCH_SIZE
    val_steps = config.VAL_SAMPLES // config.BATCH_SIZE


    train_loader = build_dataloader(
        urls=train_urls, 
        vocab=token_to_id, 
        encode_fn=vocab.encode, 
        max_len=config.TEXT_MAX_LEN,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        epoch_steps=train_steps,
        is_train=True,
        augment=True
        )
        
    val_loader = build_dataloader(
        urls=val_urls, 
        vocab=token_to_id, 
        encode_fn=vocab.encode, 
        max_len=config.TEXT_MAX_LEN,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=1,
        epoch_steps=val_steps,
        is_train=True,
        augment=False
        )


    model = CLIP(len(token_to_id), config.TEXT_MAX_LEN, config.PROJECTION_DIM).cuda(local_rank)

    for param in model.image_encoder.backbone.parameters():
        param.requires_grad = False
    model.image_encoder.backbone.eval()

    criterion = CLIPLoss().cuda(local_rank)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    # Freeze backbone weights and keep BatchNorm in eval mode

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    model = DDP(model, device_ids=[local_rank])

    best_val_loss = float('inf')

    epoch_pbar = tqdm(range(config.EPOCHS), desc="Epochs", disable=global_rank != 0)

    for epoch in epoch_pbar:

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, train_steps)

        val_loss = evaluate(model, val_loader, criterion, device, val_steps)

        if global_rank == 0:
            epoch_pbar.set_postfix({
                'Train Loss': f"{train_loss:.4f}",
                'Val Loss': f"{val_loss:.4f}"
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if global_rank == 0:
                torch.save(model.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
                tqdm.write(f"Epoch {epoch+1}: New Best Model Saved! Val Loss: {val_loss:.4f}!")
        # Only save on rank 0 to avoid file corruption from concurrent writes

        scheduler.step()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()