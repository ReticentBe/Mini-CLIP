import os
from PIL import Image
import torch
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader

def image_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


def text_transform(caption, vocab, encode_fn, max_len):
    ids, attention_mask = encode_fn(caption, vocab, max_len)
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    return (ids_tensor, mask_tensor)


def build_dataloader(urls, vocab, encode_fn, max_len, image_size, 
                     batch_size, num_workers, epoch_steps=10000):
    
    img_tf = image_transforms(image_size)
    txt_tf = lambda cap: text_transform(cap, vocab, encode_fn, max_len)

    dataset = (
        wds.WebDataset(
            urls,
            # nodesplitter=wds.split_by_node,
            shareshuffle=True,
            resampled=True
        )
        .shuffle(2000)
        .decode('pil', handler=wds.warn_and_continue)
        .to_tuple('jpg','txt')
        .map_tuple(img_tf, txt_tf)
        .map(lambda x: (x[0], x[1][0], x[1][1]))
        .batched(batch_size, partial=False)
        .with_epoch(epoch_steps)
    )

    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, 
                        pin_memory=True, prefetch_factor=2)
    
    return loader

