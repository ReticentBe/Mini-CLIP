"""
WebDataset streaming pipeline for image-text pairs.

Builds a DataLoader from .tar shards containing (jpg, json) pairs, where json
is a list of 5 captions per image. One caption is randomly selected at read 
time to avoid duplicate images within a batch
"""

import os
from PIL import Image
import torch
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader
import random

def image_transforms(image_size, is_train=True):
    """
    Build image transform pipeline

    Training: RandomResizedCrop + HorizontalFlip + ColorJitter + Normalize
    Evaluation: Resize + Normalize
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def text_transform(captions_list, vocab, encode_fn, max_len):
    """
    Randomly select one caption from the list and encode it to token IDs
    """
    caption = random.choice(captions_list)
    ids, attention_mask = encode_fn(caption, vocab, max_len)
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    return (ids_tensor, mask_tensor)


def build_dataloader(urls, vocab, encode_fn, max_len, image_size, 
                     batch_size, num_workers, epoch_steps=10000,
                     is_train=True, augment=True):
    """
    Construct a WebDataset-based DataLoader

    Args:
        urls: list of .tar shard paths
        vocab: token-to-id dictionary
        encode_fn: function(caption, vocab, max_len) -> (ids, mask)
        is_train: If True, enables resampling and shuffling
        augment: If True, applies training augmentations
    """
    
    img_tf = image_transforms(image_size, is_train=augment)
    txt_tf = lambda cap: text_transform(cap, vocab, encode_fn, max_len)

    dataset = (
        wds.WebDataset(
            urls,
            nodesplitter=wds.split_by_node,  
            resampled=is_train,                  
            handler=wds.warn_and_continue   
        )
    )
    
    if is_train:
        dataset = dataset.shuffle(2000)
    
    dataset = (
        dataset
        .decode('pil', handler=wds.warn_and_continue)
        .to_tuple('jpg','json')
        .map_tuple(img_tf, txt_tf)
        .map(lambda x: (x[0], x[1][0], x[1][1]))
        .batched(batch_size, partial=False)
    )
    
    if is_train:
        dataset = dataset.with_epoch(epoch_steps)


    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, 
                        pin_memory=True, prefetch_factor=2)
    
    return loader

