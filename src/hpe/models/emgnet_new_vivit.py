import torch
import torch.nn as nn
import time
from hpe.models.vivit import ViViT, make_vivit


class MLP(nn.Module):
    def __init__(self, infeatures=128, outfeatures=16):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(infeatures, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outfeatures))

    def forward(self, x):
        return self.mlp_head(x)


class SimCLRProjector(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimCLRProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.projector(x)


class BoundedActivation(nn.Module):
    def __init__(self, label_dim,
                 a_values=[0, -15, 0, -15, 0, -15, 0, 0, -15, 0, 0, -15, 0, 0, -15, 0],
                 b_values=[90, 15, 90, 15, 90, 15, 110, 90, 15, 110, 90, 15, 110, 90, 15, 110]):
        super(BoundedActivation, self).__init__()
        assert len(a_values) == len(
            b_values) == label_dim, "Length of a_values and b_values must be equal and equal to label_dim"
        self.a_values = nn.Parameter(torch.Tensor(a_values).view(1, -1))
        self.b_values = nn.Parameter(torch.Tensor(b_values).view(1, -1))
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(x) * (self.b_values - self.a_values) + self.a_values


class EmgNetNew(nn.Module):
    def __init__(self, proj_dim, output_size, image_size, patch_size, num_classes, num_frames, dim=192, depth=4,
                 heads=3, pool='cls', in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4):
        super(EmgNetNew, self).__init__()
        self.proj_dim = proj_dim
        self.encoder_t = ViViT(image_size, patch_size, dim, num_frames, dim, depth, heads, pool, in_channels,
                               dim_head, dropout,
                               emb_dropout, scale_dim)
        self.encoder_f = ViViT(image_size, patch_size, dim, num_frames, dim, depth, heads, pool, in_channels,
                               dim_head, dropout,
                               emb_dropout, scale_dim)

        self.projector_t = SimCLRProjector(dim, proj_dim)
        self.projector_f = SimCLRProjector(dim, proj_dim)

        self.decoder = MLP(dim * 2, output_size)
        self.bact = BoundedActivation(label_dim=output_size)

    def forward(self, x_t, x_f, return_proj=True):

        # x_t, x_f = x_t, x_f.permute(0, 2, 1) # (B, C, S)

        h_t = self.encoder_t(x_t)
        h_f = self.encoder_f(x_f)

        z_t = self.projector_t(h_t)
        z_f = self.projector_f(h_f)

        out = self.decoder(torch.cat((h_t, h_f), dim=1))
        # out = self.bact(out)

        if return_proj:
            # return h_t, h_f, z_t, z_f, out
            return out, z_t, z_f
        else:
            return out

    def load_pretrained(self, path):

        print(f'Loading pretrained model from {path}')
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
        model_dict = self.state_dict()

        pretrained_model_dict = {k: v for k, v in pretrained_dict['model_state_dict'].items() if k in model_dict}

        model_dict.update(pretrained_model_dict)
        self.load_state_dict(model_dict)
        print('Pretrained model loaded')

        return pretrained_dict['exp_setup'] if 'exp_setup' in pretrained_dict.keys() else None


def make_emgnet_new(cfg):
    emg_net_new = EmgNetNew(proj_dim=cfg.MODEL.PROJ_DIM, output_size=cfg.MODEL.OUTPUT_SIZE,
                            image_size=cfg.MODEL.IMAGE_SIZE, patch_size=cfg.MODEL.PATCH_SIZE, \
                            num_classes=cfg.MODEL.OUTPUT_SIZE, num_frames=cfg.MODEL.FRAMES, \
                            dim=cfg.TRANSFORMER.D_MODEL, heads=cfg.TRANSFORMER.NUM_HEADS, \
                            depth=cfg.TRANSFORMER.NUM_LAYERS, pool=cfg.TRANSFORMER.POOL, \
                            dim_head=cfg.TRANSFORMER.DIM_HEAD, dropout=cfg.TRANSFORMER.ATT_DROPOUT, \
                            emb_dropout=cfg.MODEL.EMB_DROPOUT,
                            scale_dim=cfg.TRANSFORMER.MLP_DIM / cfg.TRANSFORMER.D_MODEL \
                            )
    return emg_net_new


if __name__ == "__main__":
    #  test the model
    N = 2
    C = 16
    S = 200
    from hpe.config import cfg, get_param

    args = get_param(cfg, 'MODEL')
    args.update(get_param(cfg, 'TRANSFORMER'))

    model = EmgNet(**args)
    x_t = torch.rand(N, S, C)
    x_f = torch.rand(N, S, C)

    out, z_t, z_f = model(x_t, x_f)
    print(out.shape, z_t.shape, z_f.shape)

    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
