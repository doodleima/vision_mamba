import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange

from utils import pscan


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, direction='forward'):
        super(CausalConv1d, self).__init__()
        self.direction = direction
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        if self.direction == 'forward':
            x = F.pad(x, (self.padding, 0), 'constant', 0)
            x = self.conv(x)
            # x = x[:, :, :-self.padding]  # Remove padding from the end in x -> forward
        elif self.direction == 'backward':
            x = F.pad(x.flip(dims=[2]), (self.padding, 0), 'constant', 0)
            x = self.conv(x).flip(dims=[2])
            # x = self.conv(x.flip(dims=[2])).flip(dims=[2])
            # x = x[:, :, self.padding:]  # Remove padding from the beginning in fliped x -> backward
            
        return x


class CustomMambaBlock(nn.Module):
    def __init__(self, dim:int, expand_dim:int, hidden_dim:int, dt_value:int, state_dim:int):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expand_dim = expand_dim
        self.dt_value = dt_value
        self.ssm_state_dim = state_dim
        
        self.norm_layer = nn.LayerNorm((self.dim, self.hidden_dim))
        self.in_proj = nn.Linear(self.hidden_dim, self.expand_dim)     # decoder (layer for x, y)
        self.out_proj = nn.Linear(self.expand_dim, self.hidden_dim)    # encoder (layer for concat with z)

        self.fwd_conv_layer = CausalConv1d(in_channels=self.expand_dim, out_channels=self.expand_dim, direction='forward')    # forward convolution (causal)
        self.bwd_conv_layer = CausalConv1d(in_channels=self.expand_dim, out_channels=self.expand_dim, direction='backward')   # backward convolution (causal)
        # self.fwd_conv_layer = nn.Conv1d(in_channels=self.expand_dim, out_channels=self.expand_dim, kernel_size=1)
        # self.bwd_conv_layer = nn.Conv1d(in_channels=self.expand_dim, out_channels=self.expand_dim, kernel_size=1)
        self.act_layer = nn.SiLU()
        self.SSM = SSM(in_features=self.expand_dim, dt_rank=self.dt_value, dim_inner=self.expand_dim, d_state=self.ssm_state_dim)   # SSM (from https://github.com/kyegomez/VisionMamba)
        
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x1 = self.norm_layer(x)
        xy = self.in_proj(x1)    # x and y after decoded        
        z_out = self.act_layer(xy)
        # z = rearrange(self.act_layer(xy), "b sl c -> b c sl")

        xy = rearrange(xy, "b sl c -> b c sl")    # conv1d expected input: batch size, channels, sequence length
        x1 = rearrange(self.act_layer(self.fwd_conv_layer(xy)), "b c sl -> b sl c")
        y = rearrange(self.act_layer(self.bwd_conv_layer(xy)), "b c sl -> b sl c")

        ### SSM        
        x_out = self.SSM(x1)
        y_out = self.SSM(y)
        
        ### matmul
        xy_out = (x_out*z_out) + (y_out*z_out)
        xy_enc = self.out_proj(xy_out)        

        xy_final = x + xy_enc

        return xy_final


def selective_scan(x, delta, A, B, C, D):
    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

    hs = pscan(deltaA, BX)

    y = (hs @ C.unsqueeze(-1)).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
    y = y + D * x

    return y


def selective_scan_seq(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

    h = torch.zeros(x.size(0), dim_inner, d_state, device=deltaA.device)  # (B, ED, N)
    hs = []

    for t in range(0, L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

    # y = (C.unsqueeze(2) * hs).sum(3)
    y = (hs @ C.unsqueeze(-1)).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
    y = y + D * x

    return y


class SSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):
        super(SSM, self).__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(dim_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # x_reshaped = x.reshape(-1, 768)
        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj_layer(delta))

        # Assuming selective_scan and selective_scan_seq are defined functions
        if pscan: y = selective_scan(x, delta, A, B, C, D)
        else: y = selective_scan_seq(x, delta, A, B, C, D)

        return y