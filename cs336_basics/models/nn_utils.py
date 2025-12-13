from jaxtyping import Float,Int
from torch import Tensor
import torch

def log_softmax(x:Float[Tensor,"..."],dim:int):
    
    # compute the log_sum_exp
    x_max       = torch.max(input = x,dim = dim, keepdim=True).values
    x_reduced   = x - x_max
    log_sum_exp = (torch.exp(x_reduced).sum(dim = dim, keepdim=True)).log() + x_max

    log_softmax = x - log_sum_exp
    return log_softmax 


def cross_entropy(logits: Float[Tensor, "... seq_len vocab_size"], targets: Int[Tensor, "seq_len"]):
    logps = log_softmax(x = logits,dim = -1)
    
    target_logps = torch.gather(
        input   = logps,
        dim     = -1,
        index   = targets.unsqueeze(-1)
    )   # [batch_size, seq_len, 1]
    
    loss = -target_logps.mean()
    return loss

def softmax(x: Float[Tensor, "... d_model"], dim: int):
    x_reduced = x - torch.max(x, dim = dim, keepdim = True).values
    x_exp     = torch.exp(x_reduced)
    return x_exp / x_exp.sum(dim = dim, keepdim=True)