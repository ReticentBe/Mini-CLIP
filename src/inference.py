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

    transform = image_transforms(config.IMAGE_SIZE)
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
        print("Vocabulary is not exists")
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
    
    test_image_path = os.path.join(config.DATA_PATH, "")

    candidates = [
        "a dog running on the grass",                             
        "two men playing football",                               
        "a black car parked on the street",                       
        "a little girl going into a wooden building"              
    ]

    predict(test_image_path, candidates, model, token_to_id, device)

if __name__ == "__main__":
    main()
