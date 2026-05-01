import json
import pathlib


# tokenize输入文本为字符串，采用split划分为单词组成的列表，以单词作为token
# build_vocabulary接受文本作为字符串输入，目标是建立token到id的映射字典，方便后续以id进行embedding。包含PAD,UNK,BOS,EOS四个特殊token
# encode将输入的文本根据字典映射为id，大于max_len的进行截断，小于max_len的采用PAD填充，并在首尾添加起始符BOS和终止符EOS，并保存相应的有效文本在掩码中，方便后续transformer
# save_vocab将计算好的字典保存，并可以使用load_vocab读取

def tokenize(text):
    text = text.lower()
    for p in ['.', ',', '!', '?', '"', "'"]:
        text = text.replace(p, ' ')
    tokens = text.split()  
# 按照空格将字符串切分，更好的实现方法应该是使用正则表达式，可以直接不提前replace处理
    return tokens

def build_vocabulary(captions):
    token_to_id = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
    id_to_token = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    count = 4
    for caption in captions:
        for token in tokenize(caption):
            if token not in token_to_id.keys():
                token_to_id[token] = count
                id_to_token.append(token)
# 若使用list.index(token)来进行编号，不使用token_to_id，则每次都需要从头到尾的线性扫描，效率远不如dict
                count += 1

    return token_to_id, id_to_token
        

def encode(caption, token_to_id, max_len):
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
        json.dump(token_to_id, f)  # 保存为json格式


def load_vocab(path):
    with open(path, 'r') as f:
        token_to_id = json.load(f)
    return token_to_id
