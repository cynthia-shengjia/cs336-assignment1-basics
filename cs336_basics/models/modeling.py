import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device = None, dtype = None):
        super().__init__()
        
        self.sigma  = 2/(in_features + out_features)
        self.weight = nn.Parameter(torch.empty(out_features,in_features,device = device,dtype = dtype))
        self._init_weight()

    def _init_weight(self):
        nn.init.trunc_normal_(
            tensor  = self.weight,
            mean    = 0,
            std     = self.sigma,
            a       = -3 * self.sigma,
            b       =  3 * self.sigma
        )
    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]):
        return x @ self.weight.T 

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
    def _init_weight(self):
        nn.init.trunc_normal_(
            tensor = self.weight,
            mean = 0,
            std  = 1,
            a = -3,
            b = 3
        )
    def forward(self, token_ids: Float[Tensor,"batch_size seq_len"]):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device =  device,dtype = dtype))
        self.eps    = eps
    def _init_weight(self):
        pass
    def forward(self,x:Float[Tensor, "batch_size seq_len d_model"]):
        decay = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim= True) + self.eps)      # [B,L,1]
        return x * self.weight / decay


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device = None, dtype = None):
        super().__init__()
        self.activation = lambda x: x * torch.sigmoid(x)
        self.w1 = Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)
        self.w2 = Linear(in_features=d_ff,out_features=d_model,device=device,dtype=dtype)
        self.w3 = Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)
    def forward(self,x:Float[Tensor,"batch_size seq_len d_model"]):
        gate         = self.activation(self.w1(x))
        hidden       = self.w3(x)
        gated_hidden = gate * hidden
        return self.w2(gated_hidden)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None,dtype = None):
        super().__init__()
        self.theta       = theta 
        self.d_k         = d_k
        self.max_seq_len = max_seq_len
        
        self.register_buffer(
            name   = "rope_transformation",
            tensor = self._construct_rope(),
            persistent = True
        )
    
    def _construct_rope(self):
        position   = torch.arange(start = 0, end = self.max_seq_len, step = 1).unsqueeze(dim = 1)          # [max_seq_len, 1]
        freqs      = 1 / self.theta ** ( torch.arange(start = 0, end = self.d_k, step = 2) / self.d_k)     # [1, d_k/2]
        rope_theta = position * freqs                                                                      # [max_seq_len, d_k/2]
        return rope_theta

    def forward(self,x: Float[Tensor, "... seq_len d_model"], token_positions: Float[Tensor, "... seq_len"]):
        iter_leave_input = rearrange(x, "... (x y) -> ... x y", y = 2)  # [... seq_len d_k/2 2]
        seq_theta = self.rope_transformation[token_positions]               # [max_seq_len,d_k/2] -> [seq_len,d_k/2]
        
        cos = seq_theta.cos()
        sin = seq_theta.sin()

        x1 = iter_leave_input[...,0] # [... seq_len d_k/2]
        x2 = iter_leave_input[...,1] # [... seq_len d_k/2]

        feature1 = cos * x1 - sin * x2          # [... seq_len d_k/2]
        feature2 = sin * x1 + cos * x2          # [... seq_len d_k/2]

        return torch.stack([feature1,feature2], dim = -1).flatten(start_dim=-2,end_dim=-1)



        
        
        
        
        