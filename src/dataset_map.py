import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class CLIPDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, encode, max_len, transform=None):
        '''
        初始化Dataset
        读取captions_file, 把所有的 (image_path, captions)对存到一个列表里
        image_dir: 图片所在的文件夹路径
        captions_file: 有图片文本对应关系的文本路径
        vocab: token to id字典
        max_len: 文本最大长度, 在config定义
        transform: torchvision transforms
        '''
        super().__init__()
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.encode = encode
        self.max_len = max_len
        self.transform = transform

        '''
        self.samples = []    # (image_path, captions)

        with open(captions_file, 'r', encoding='utf-8') as f:
            next(f) # 跳过image, caption的表头
            for line in f:
                line = line.strip()  # 去除末尾的换行符 \n
                if not line:
                    continue   # 跳过空行
                parts = line.split(',', 1)   # 按逗号切分，最多切分一次，防止对caption切分
                if len(parts) == 2:
                    image_name = parts[0]
                    captions = parts[1]
                    self.samples.append((image_name, captions))
        '''

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        '''
        根据索引从数据集中返回一个样本
        对该样本的图片进行transform, 文本进行encode
        '''
        image_path = self.image_paths[index]
        caption = self.captions[index]

        # 读取图片并转化为RGB格式
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)  

        # 文本预处理
        ids, attention_mask = self.encode(caption, self.vocab, self.max_len)
        
        ids_tensor = torch.tensor(ids, dtype=torch.long)    
        # nn.Embedding要求张量必须是torch.long, 为健壮性因此dtype操作
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        # 与ids_tensor保持一致

        return image, ids_tensor, mask_tensor


def image_transforms(image_size):     # 由于模态同步，翻转可能会导致图片和文本描述不匹配，因此不做翻转处理
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(                  # 使用 ImageNet 的标准均值和方差对RGB三个维度进行归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


        
            



