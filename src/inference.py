"""
Inference script for image-text retrieval demo

Loads a trained CLIP checkpoint and runs two demos:
1. In-distribution: Flickr8k image with ground-truth and distractor captions
2. Out-distribution: Lenna portrait with manually crafted captions
"""


import os
import torch
import torch.nn.functional as F
from PIL import Image

import config
import vocab
from dataset import image_transforms
from model import CLIP

def predict(image_path, text_candidates, model, token_to_id, device):
    """
    image_path: path of an image
    text_candidates: a List contains several text descriptions, e.g. ['a dog', 'a cat']
    """
    model.eval()

    transform = image_transforms(config.IMAGE_SIZE, is_train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    all_ids = []
    all_masks = []
    for text in text_candidates:
        ids, masks = vocab.encode(text, token_to_id, config.TEXT_MAX_LEN)
        all_ids.append(ids)
        all_masks.append(masks)

    ids_tensor = torch.tensor(all_ids, dtype=torch.long).to(device)
    masks_tensor = torch.tensor(all_masks, dtype=torch.long).to(device) 
    # (N, max_len)

    with torch.no_grad():
        image_features = model.image_encoder(image_tensor)  # (1, Projection_dim)
        text_features = model.text_encoder(ids_tensor, masks_tensor)  # (N, Projection_dim)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = (image_features @ text_features.T).squeeze(0) #(N, )

        logits_scale = torch.exp(model.logit_scale)
        probs = (logits_scale * similarity).softmax(dim=-1)

        scores = probs.cpu().numpy()

    results = sorted(zip(text_candidates, scores), key=lambda x:x[1], reverse=True)
    for i, (text, score) in enumerate(results):
        print(f'Top {i+1}: [{score:.4f}] {text}')


def main():
    device = config.DEVICE
    if not os.path.exists(config.VOCAB_PATH):
        print("Vocabulary file not found")
        return
    else:
        token_to_id = vocab.load_vocab(config.VOCAB_PATH)

    model = CLIP(len(token_to_id), config.TEXT_MAX_LEN, config.PROJECTION_DIM).to(device)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Successfully loaded best_model.pth")
    else:
        print("Can't find best_model.pth")
        return
    

    print("=" * 50)
    print("Demo 1: Flickr8k (In-Distribution)")
    print("=" * 50)
    flickr_image = os.path.join(config.DATA_PATH, "69189650_6687da7280.jpg")
    flickr_candidates = [
        "A brown dog is running through a brown field .",          
        "A cat sleeping on a couch",           
        "A dog sitting in the snow",           
        "A child running on a brown field",        
        "A dog running through a field",       
    ]

    predict(flickr_image, flickr_candidates, model, token_to_id, device)

    print("\n" + "=" * 50)
    print("Demo 2: Lenna (Out-of-Distribution)")
    print("=" * 50)
    lenna_image = os.path.join(config.DATA_PATH, "Lenna.jpg")
    lenna_candidates = [
        "A woman wearing a hat with feathers",
        "A man wearing a hat with feathers", 
        "A woman without any hat",
        "A child looking over her shoulder",
        "A portrait of an elderly woman",
    ]

    predict(lenna_image, lenna_candidates, model, token_to_id, device)


if __name__ == "__main__":
    main()
