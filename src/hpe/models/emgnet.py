import torch
import torch.nn as nn
import time
from hpe.models.transformer import VViT, make_vvit


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
                 a_values = [0, -15, 0, -15, 0, -15, 0, 0, -15, 0, 0, -15, 0, 0, -15, 0], 
                 b_values = [90, 15 , 90, 15, 90, 15, 110, 90, 15, 110, 90, 15, 110, 90, 15, 110]):
        super(BoundedActivation, self).__init__()
        assert len(a_values) == len(b_values) == label_dim, "Length of a_values and b_values must be equal and equal to label_dim"
        self.a_values = nn.Parameter(torch.Tensor(a_values).view(1, -1))
        self.b_values = nn.Parameter(torch.Tensor(b_values).view(1, -1))
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(x) * (self.b_values - self.a_values) + self.a_values
    

class EmgNet(nn.Module):
    def __init__(self, proj_dim, output_size, **kwargs):
        super(EmgNet, self).__init__()
        self.proj_dim = proj_dim
        self.encoder_t = VViT(**kwargs)
        self.encoder_f = VViT(**kwargs)

        self.projector_t = SimCLRProjector(self.encoder_t.d_model, proj_dim)
        self.projector_f = SimCLRProjector(self.encoder_f.d_model, proj_dim)

        self.decoder = MLP(self.encoder_t.d_model+self.encoder_f.d_model, output_size)
        self.bact = BoundedActivation(label_dim=output_size)
    
    def forward(self, x_t, x_f, return_proj=True):

        # x_t, x_f = x_t, x_f.permute(0, 2, 1) # (B, C, S)

        h_t = self.encoder_t(x_t)
        h_f = self.encoder_f(x_f)

        z_t = self.projector_t(h_t)
        z_f = self.projector_f(h_f)

        out = self.decoder(torch.cat((h_t,h_f), dim=1))
        # out = self.bact(out)

        if return_proj:
            # return h_t, h_f, z_t, z_f, out
            return out, z_t, z_f
        else:
            return out
    
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