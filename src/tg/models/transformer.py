# Implementation of video ViT model adopted from 

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

 

# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, return_attn=False):
        
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x)
            x = ff_layer(x)
        return x


class VViT(nn.Module):
    __acceptable_attributes__ = ["input_size", "input_patch_size", "frames", 
                                 "frame_patch_size", "num_classes", "d_model", "num_layers", 
                                 "num_heads", "mlp_dim", "pool" , "channels" , "dim_head", 
                                 "att_dropout" , "emb_dropout"]
    
    def __init__(self, **kwargs):
        super().__init__()
        [self.__setattr__(k, kwargs.get(k)) for k in self.__acceptable_attributes__]

        image_height, image_width = pair(self.input_size)
        patch_height, patch_width = pair(self.input_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert self.frames % self.frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (self.frames // self.frame_patch_size)
        patch_dim = self.channels * patch_height * patch_width * self.frame_patch_size

        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = self.frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.dropout = nn.Dropout(self.emb_dropout)

        self.transformer = Transformer(self.d_model, self.num_layers, self.num_heads, self.dim_head, self.mlp_dim, self.att_dropout)

        self.to_latent = nn.Identity()

        # self.decoder = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes),
        # )

    def forward(self, x):

        #  input x: (batch_size, frames, C)
        x = x.reshape(x.shape[0], 1, x.shape[1], 4, 4)
        #  shape of x: (batch_size, 1, frames, height, width)
        x = self.to_patch_embedding(x) # (batch_size, n, dim) where n = fram_path_size*patch_size**2
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] # (batch_size, n+1, dim)
        x = self.dropout(x)

        x = self.transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x
    
    def load_pretrained(self, path):

        print(f'Loading pretrained model from {path}')
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        print('Pretrained model loaded')


def make_vvit(cfg):
    from tg.config import get_model_param
    args = get_model_param(cfg)
    return VViT(**args)

def vis_atten(module, input, output):
    global attn_before_softmax, attn_after_softmax
    attn_before_softmax.append(input[0].detach().cpu())
    attn_after_softmax.append(output.detach().cpu())
    
    

if __name__ == "__main__":

    model = make_vvit(cfg=None)
    # Generate random input data

    N = 2  # Number of training examples
    S = 200   # Sequence length
    C = 16   # Number of channels
    input_data = torch.randn(N, S, C)
    out = model(input_data)
    
    print(out.shape)