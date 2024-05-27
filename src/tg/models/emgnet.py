import torch
import torch.nn as nn
import time
from tg.models.transformer import VViT, make_vvit


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
    def __init__(self, input_size, proj_dim, d_model, seq_length, output_size, mlp_dim, ):
        super(EmgNet, self).__init__()
        self.proj_dim = proj_dim
        self.d_model = d_model
        self.seq_length = seq_length
        self.encoder_t = VViT(
                            image_size = 4,
                            image_patch_size = 2,
                            frames = seq_length,
                            frame_patch_size = 4,
                            num_classes = output_size,
                            dim = 128,
                            depth = 4,
                            heads = 4,
                            mlp_dim = 2048,
                            dropout = 0.25,
                            emb_dropout = 0.5,
                            channels=1,
                        )
        self.encoder_f = VViT(
                            image_size = 4,
                            image_patch_size = 2,
                            frames = seq_length,
                            frame_patch_size = 4,
                            num_classes = output_size,
                            dim = 128,
                            depth = 4,
                            heads = 4,
                            mlp_dim = 2048,
                            dropout = 0.25,
                            emb_dropout = 0.5,
                            channels=1,
                        )        
        self.projector_t = SimCLRProjector(self.d_model, proj_dim)
        self.projector_f = SimCLRProjector(self.d_model, proj_dim)

        self.decoder = MLP(self.d_model*2, output_size)
        self.bact = BoundedActivation(label_dim=output_size)
    
    def forward(self, x_t, x_f):

        # x_t, x_f = x_t, x_f.permute(0, 2, 1) # (B, C, S)

        h_t = self.encoder_t(x_t)
        h_f = self.encoder_f(x_f)

        z_t = self.projector_t(h_t)
        z_f = self.projector_f(h_f)

        out = self.decoder(torch.cat((h_t,h_f), dim=1))
        # out = self.bact(out)


        # return h_t, h_f, z_t, z_f, out
        return out, z_t, z_f
    
if __name__ == "__main__":

    #  test the model
    N = 2
    C = 16
    S = 200

    model = EmgNet(input_size=C, proj_dim=128, seq_length=S, output_size=16, d_model=128, mlp_dim=2048)
    x_t = torch.rand(N, S, C)
    x_f = torch.rand(N, S, C)

    x_t, x_f, z_t, z_f = model(x_t, x_f)
    print(x_t.shape, x_f.shape, z_t.shape, z_f.shape)
    print(model)

    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)