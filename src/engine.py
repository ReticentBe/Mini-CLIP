"""
Training and Evaluation loops

train_one_epoch: Standard DDP training with frozen backbone eval() mode
evaluate: Accumulates embeddings across GPUs via all_gather, computes global 
N to N similarity matrix and reports Recall@1/5/10.
"""

import torch
from tqdm import tqdm
import torch.distributed as dist

def train_one_epoch(model, dataloader, optimizer, criterion, device, total_steps=None):
    """
    Run one training epoch, returning average loss

    Note: backbone.eval() is called after model.train() to freeze BatchNorm 
    running statistics in the pretrained image encoder.
    """

    model.train()

    if hasattr(model, 'module'):
        model.module.image_encoder.backbone.eval()
    else:
        model.image_encoder.backbone.eval()
    
    total_loss = 0.0
    is_main_process = dist.get_rank() == 0
    pbar = tqdm(dataloader, desc="Training", leave=False, total=total_steps, disable=not is_main_process)

    total_samples = 0

    for step, batch in enumerate(pbar):
        if total_steps is not None and step >= total_steps:
            break
        images, input_ids, attention_mask = batch
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        image_embeddings, text_embeddings = model(images, input_ids, attention_mask)
        logit_scale = model.module.logit_scale.exp() if hasattr(model, 'module') else model.logit_scale.exp()

        loss = criterion(image_embeddings, text_embeddings, logit_scale)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]

        pbar.set_postfix({'loss': loss.item()})

        total_samples += images.shape[0]
    
    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, total_steps=None):
    """
    Evaluate model and compute Image-to-Text Recall@1/5/10

    All per-GPU embeddings are gathered via all_gather, then a global 
    similarity matrix is used for ranking-based retrieval metrics
    """
    model.eval()
    total_loss = 0.0
    all_image_features = []
    all_text_features = []

    is_main_process = dist.get_rank() == 0
    pbar = tqdm(dataloader, desc="Evaluation", leave=False, total=total_steps, disable=not is_main_process)

    total_samples = 0
    for step, batch in enumerate(pbar):
        if total_steps is not None and step >= total_steps:
            break
        
        image, input_ids, attention_mask = batch

        image = image.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        image_embeddings, text_embeddings = model(image, input_ids, attention_mask)

        logit_scale = model.module.logit_scale.exp() if hasattr(model, 'module') else model.logit_scale.exp()

        loss = criterion(image_embeddings, text_embeddings, logit_scale)
        
        total_loss += loss.item() * image.shape[0]

        all_image_features.append(image_embeddings)
        all_text_features.append(text_embeddings)

        total_samples += image.shape[0]

        pbar.set_postfix({'val_loss': loss.item()})

    all_image_features = torch.cat(all_image_features, dim=0)  # (N, dim)
    all_text_features = torch.cat(all_text_features, dim=0)  # (N, dim)

    global_image_features_list = [torch.zeros_like(all_image_features) for _ in range(dist.get_world_size())]
    global_text_features_list = [torch.zeros_like(all_text_features) for _ in range(dist.get_world_size())]

    dist.all_gather(global_image_features_list, all_image_features)
    dist.all_gather(global_text_features_list, all_text_features)

    global_image_features = torch.cat(global_image_features_list, dim=0)
    global_text_features = torch.cat(global_text_features_list, dim=0)

    similarity_matrix = global_image_features @ global_text_features.T # (N, N)

    _, indices = torch.topk(similarity_matrix, k=10, dim=-1)  # indice: (N, K)
    N_global = similarity_matrix.shape[0]
    targets = torch.arange(N_global, device=similarity_matrix.device).reshape(-1, 1) # (N, 1)

    matches = (indices == targets) # (N, K) True/False 

    recall_1 = matches[:, :1].sum().item() / N_global
    recall_5 = matches[:, :5].sum().item() / N_global
    recall_10 = matches[:, :10].sum().item() / N_global

    if dist.get_rank() == 0:
        print(f"I2T Recall@1: {recall_1:.4f}, Recall@5: {recall_5:.4f}, Recall@10: {recall_10:.4f}")

    
    return total_loss / total_samples