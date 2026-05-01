import torch
from torch import nn
import torchvision.models as models
import math

class ImageEncoder(nn.Module):
    def __init__(self, backbone='resnet18', projection_dim=512):
        super().__init__()
        model_func = getattr(models, backbone)  # model.backbone
        self.backbone = model_func(weights='DEFAULT')
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  
        self.projection = nn.Linear(feature_dim, projection_dim)  
        # 不需要对图片进行分类，因此将最后一个fc层替换为Identity层，保持原有特征维度
        # 并且引入Identity可以解耦backbone和head
        # 定义一个projection head，将原本最后一层的输入维度映射到peojection_dim

    def forward(self, x):
        return self.projection(self.backbone(x))
    # in: (batch_size, 3, H, W)
    # out: (batch_size, projection_dim)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels=in_channels, out_channels=embed_dim, 
            kernel_size=patch_size, stride=patch_size
            )
        
    def forward(self, x):
        x = self.projection(x)  
        # (B, in_channels, H, W) -> (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  
        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)
        # (B, num_patches, embed_dim)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, 
                 num_heads=12, num_layers=12, projection_dim=512):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.clstoken = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
        self.projection = nn.Linear(embed_dim, projection_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.clstoken.expand(B, -1, -1) # (B, 1, embed_dim)

        x = torch.cat((cls_token, x), dim=1) # (B, num_patches+1, embed_dim)

        x = x + self.pos_embed
        
        x = self.transformer(x)

        cls_output = x[:, 0, :]  # (B, embed_dim)

        return self.projection(cls_output)  # (B, projection_dim)





class PositionEncoder(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        P = torch.zeros((1, max_len, model_dim))
        X = torch.arange(max_len, dtype=torch.float).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, model_dim, 2, dtype=torch.float) / 
            model_dim
        )

        P[:,:,0::2] = torch.sin(X)
        P[:,:,1::2] = torch.cos(X)
        self.register_buffer('P', P)
    
    def forward(self, x):
        return self.P[:, :x.shape[1], :]
        


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, model_dim=512, num_heads=8, num_layers=12, projection_dim=512):
        super().__init__()
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        # Position Encoding
        self.position_encode = PositionEncoder(model_dim, max_len)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(model_dim, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Projection
        self.projection = nn.Linear(model_dim, projection_dim)

    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        # (batch_size, seq_len) -> (batch_size, seq_len, model_dim)
        x = x + self.position_encode(x)  
        padding_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # (batch_size, seq_len, model_dim)
        pooled_x = x[:,0,:]
        # (batch_size, model_dim)
        return self.projection(pooled_x)
        # (batch_size, projection_dim)
    


class CLIP(nn.Module):
    def __init__(self, vocab_size, max_len, projection_dim=512, init_temp=0.07):
        super().__init__()
        self.image_encoder = VisionTransformer(projection_dim=projection_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, max_len=max_len, projection_dim=projection_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # log ensure τ>0

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)  # (batch, projection_dim)
        text_features = self.text_encoder(input_ids, attention_mask) # (batch, projection_dim)
        # L2 Normalize
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        text_features = nn.functional.normalize(text_features, p=2, dim=1)

        return image_features, text_features











