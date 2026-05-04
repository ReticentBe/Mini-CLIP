"""
Vocabulary builder and text tokenizer

Provides word-level tokenization, vocabulary construction with special tokens: 
<PAD>, <UNK>, <BOS>, <EOS>, encoding with padding/truncation, and vocabulary
storage via JSON.

"""

import json
import pathlib

def tokenize(text):
    """Lowercase and split text into word tokens, stripping punctuation"""
    text = text.lower()
    for p in ['.', ',', '!', '?', '"', "'"]:
        text = text.replace(p, ' ')
    tokens = text.split()  
    # Note: regex-based tokenizer would be more robust
    return tokens

def build_vocabulary(captions):
    """
    Build token-to-id mapping from a list of caption strings

    Returns:
        tuple: (token_to_id dict, id_to_token list)
    """
    token_to_id = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
    id_to_token = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    count = 4
    for caption in captions:
        for token in tokenize(caption):
            if token not in token_to_id.keys():
                token_to_id[token] = count
                id_to_token.append(token)
                count += 1

    return token_to_id, id_to_token
        

def encode(caption, token_to_id, max_len):
    """
    Encode a caption string into padded token IDs and attention mask

    Prepends <BOS>, appends <EOS>, truncates or pads to max_len

    Returns:
        tuple: (ids list[int], attention_mask[int])
    """
    max_content_len = max_len - 2
    pad_id = token_to_id["<PAD>"]
    unk_id = token_to_id["<UNK>"]
    tokens = tokenize(caption)
    if len(tokens) > max_content_len:
        tokens = tokens[:max_content_len]
    tokens = ["<BOS>"] + tokens + ["<EOS>"]
    if len(tokens) < max_len:
        tokens = tokens + (max_len-len(tokens)) * ["<PAD>"]
    ids = []
    attention_mask = []
    for token in tokens:
        id = token_to_id.get(token, unk_id)
        ids.append(id)
        if id != pad_id:
            attention_mask.append(1)
        else: 
            attention_mask.append(0)
    return ids, attention_mask
    

def save_vocab(token_to_id, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(token_to_id, f) 


def load_vocab(path):
    with open(path, 'r') as f:
        token_to_id = json.load(f)
    return token_to_id
