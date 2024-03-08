import os
# from pathlib import Path
# sys.path.append(str(Path.cwd()))

from layer import CustomMambaBlock
from utils import VisionRotaryEmbeddingFast

import torch
import torch.nn as nn

# from collections.abc import Sequence
from typing import Sequence, Union
from monai.utils import deprecated_arg
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock

class CustomVimBlock(nn.Module):
    @deprecated_arg(name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead.")
    def __init__(self,
                in_channels: int,
                img_size: Union[Sequence[int], int], #Sequence[int] | int,
                patch_size: Union[Sequence[int], int], #Sequence[int] | int,
                hidden_dim: int,                # hidden_size: int = 768,
                expand_dim: int, 
                mlp_dim: int,                   # for mlp layer
                num_layers: int,                # number of mamba enc
                num_heads: int,                 # number of head?
                num_classes: int,               # output channel
                dropout_rate: float,
                spatial_dims: int,
                proj_type: str = "conv",
                dt_value: int = 4,              # param for delta value calucation
                ssm_state_dim: int = 16,        # ssm dimension (in paper default 16)
                is_rope: bool = False,
                # act_function: str = "silu",
                # # pos_embed: str = "conv",
                # # pos_embed_type: str = "learnable",
                # # classification: bool = False,
                ) -> None:
    
        super().__init__()

        if not (0 <= dropout_rate <= 1): raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        if (img_size//patch_size)**spatial_dims != hidden_dim: 
            print(f'Patches HWD: {img_size//patch_size}, Dimension: {hidden_dim}')
            raise ValueError("hidden_dim should be equal with patch_size^(spatial_dims)")
            
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            # proj_type=proj_type,
            pos_embed=proj_type,
            # pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.mamba = CustomMambaBlock(dim=hidden_dim+self.cls_token.shape[0], hidden_dim=hidden_dim, expand_dim=expand_dim, dt_value=dt_value, state_dim=ssm_state_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)
        
        ### 1. MONAI based - multiple block structure: (mamba -> mlp) * n
        ### 2. COMMON approach - multiple mamba & single mlp strucutre: (mamba) * n -> mlp
        
        self.mamba_enc = nn.ModuleList([self.mamba for i in range(num_layers)])
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, num_classes))

        # self.mamba_block = nn.Sequential(self.mamba, self.mlp)
        # self.mamba_enc = nn.ModuleList([self.mamba_block for i in range(num_layers)])
        # self.classifier = nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.Tanh())
                
        
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.is_rope: x = self.rope(x)
        x = torch.cat((cls_token, x), dim=1)
        for block in self.mamba_enc: x = block(x) # x = self.mamba(x)
        x = self.mlp(x)
        x_out = self.classifier(x[:,0]) # logit
        
        return x_out