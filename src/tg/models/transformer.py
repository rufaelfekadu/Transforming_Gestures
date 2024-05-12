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

class BoundedActivation(nn.Module):
    def __init__(self, label_dim, 
                 a_values = [0, -15, 0, -15, 0, -15, 0, 0, -15, 0, 0, -15, 0, 0, -15, 0], 
                 b_values = [90, 15 , 90, 15, 90, 15, 110, 90, 15, 110, 90, 15, 110, 90, 15, 110]):
        super(BoundedActivation, self).__init__()
        assert len(a_values) == len(b_values) == label_dim, "Length of a_values and b_values must be equal and equal to label_dim"
        self.a_values = nn.Parameter(torch.Tensor(a_values).view(1, -1))
        self.b_values = nn.Parameter(torch.Tensor(b_values).view(1, -1))
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(x) * (self.b_values - self.a_values) + self.a_values
    
class VViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            # BoundedActivation(num_classes)
        )

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
        return self.decoder(x)
    
    def load_pretrained(self, path):

        print(f'Loading pretrained model from {path}')
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        print('Pretrained model loaded')


def make_vvit(cfg):
    return VViT(
        image_size = 4,
        image_patch_size = 2,
        frames = cfg.DATA.SEGMENT_LENGTH,
        frame_patch_size = 4,
        num_classes = len(cfg.DATA.LABEL_COLUMNS),
        dim = 128,
        depth = 4,
        heads = 4,
        mlp_dim = 2048,
        dropout = 0.25,
        emb_dropout = 0.5,
        channels=1,
    )

def vis_atten(module, input, output):
    global attn_before_softmax, attn_after_softmax
    attn_before_softmax.append(input[0].detach().cpu())
    attn_after_softmax.append(output.detach().cpu())
    
    

if __name__ == "__main__":
    from hpe.config import cfg
    model = make_vvit(cfg)
    # register hook on attention module
    model.transformer.layers[-1][0].attend.register_forward_hook(vis_atten)
    print(model)
    # Generate random input data

    N = 2  # Number of training examples
    S = 150   # Sequence length
    C = 16   # Number of channels
    input_data = torch.randn(N, S, C)
    model.eval()
    out = model(input_data)
    
    print(out.shape)

    #  plot attention
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set()
    attn_before_softmax = attn_before_softmax[0].squeeze(0)[0,:,0,1:].mean(axis=0)
    attn_after_softmax = attn_after_softmax[0].squeeze(0)[0,:,0,1:].mean(axis=0)

    # reshape to 4, 125
    attn_before_softmax = attn_before_softmax.reshape(-1, 75)
    # attn_after_softmax = attn_after_softmax.reshape(4, 125)

    plt.figure(figsize=(10, 5))
    # plt.grid(False)

    plt.subplot(1, 2, 1)
    plt.imshow(attn_before_softmax, aspect='auto', cmap='inferno')
    plt.title('Attention before softmax')

    # interpolate to shape 16,250 with nearest interpolation
    # attn_before_softmax = np.repeat(attn_before_softmax, 2, axis=0)
    # attn_before_softmax = np.repeat(attn_before_softmax, 2, axis=1)
    # attn_before_softmax = np.repeat(attn_before_softmax, 2, axis=2)

    #  interpolate using torch
    attn_before_softmax_inter = F.interpolate(attn_before_softmax.unsqueeze(0).unsqueeze(0), scale_factor=(4, 2), mode='nearest').numpy()

    plt.subplot(1, 2, 2)
    plt.imshow(attn_before_softmax_inter.reshape(-1,150), aspect='auto', cmap='inferno')
    plt.title('Attention before softmax')

    # # plot input data
    # plt.subplot(1, 2, 2)
    # plt.imshow(input_data[0,:,:].reshape(16,150), aspect='auto', cmap='hot')
    #  remove grid
    # fig, axs = plt.subplots(1, attn_before_softmax.shape[0], figsize=(10, 10))
    # for i in range(attn_before_softmax.shape[0]):
    #     axs[i].imshow(attn_before_softmax[i,:].reshape(4, 125), aspect='auto')
    #     axs[i].set_title(f'Attention before softmax for head {i}')
    
    plt.show()