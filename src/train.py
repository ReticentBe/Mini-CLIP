import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import nn
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
                vocab.build_vocabulary(all_captions)
    dist.barrier()
    
    token_to_id = vocab.load_vocab(config.VOCAB_PATH)

    shards_pattern = os.path.join(config.WEBDATASET_DIR, "*.tar")
    all_tars = sorted(glob.glob(shards_pattern))

    if len(all_tars) == 0:
        raise ValueError(f"No tar files found in {config.WEBDATASET_DIR}!")
    
    train_urls = all_tars[:-4]
    val_urls = all_tars[-4:]  

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
        epoch_steps=train_steps
        )
        
    val_loader = build_dataloader(
        urls=val_urls, 
        vocab=token_to_id, 
        encode_fn=vocab.encode, 
        max_len=config.TEXT_MAX_LEN,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        epoch_steps=val_steps
        )


    model = CLIP(len(token_to_id), config.TEXT_MAX_LEN, config.PROJECTION_DIM).cuda(local_rank)
    criterion = CLIPLoss().cuda(local_rank)
    optimizer = optim.AdamW(
        model.parameters(),
        lr = config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    model = DDP(model, device_ids=[local_rank])

    best_val_loss = float('inf')

    epoch_pbar = tqdm(range(config.EPOCHS), desc="Epochs", disable=global_rank != 0)

    for epoch in epoch_pbar:

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss = evaluate(model, val_loader, criterion, device)

        if global_rank == 0:
            epoch_pbar.set_postfix({
                'Train Loss': f"{train_loss:.4f}",
                'Val Loss': f"{val_loss:.4f}"
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            if global_rank == 0:
                tqdm.write(f"Epoch {epoch+1}: New Best Model Saved! Val Loss: {val_loss:.4f}!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()