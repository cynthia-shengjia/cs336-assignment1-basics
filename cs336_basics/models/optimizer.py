from collections.abc import Callable, Iterable 
from typing import Optional,Tuple
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,
        params: Iterable          = None,
        lr:float                  = 1e-5, 
        betas: Tuple[float,float] = (0.9,0.999),
        weight_decay: float       = 1e-3,
        eps: float                = 1e-5
    ):
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "wd":    weight_decay,
            "eps":   eps
        }
        super().__init__(
            params      = params,
            defaults    = defaults
        )
    def step(self, closure: Callable = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:    
            lr      = group["lr"]
            beta1   = group["beta1"]
            beta2   = group["beta2"]
            wd      = group["wd"]
            eps     = group["eps"]

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]       # get the state of the parameters
                grad = p.grad.data          # compute the gradient of the model parameter

                m    = state.get("m", torch.zeros_like(grad))     
                v    = state.get("v", torch.zeros_like(grad))
                t    = state.get("t", 1)

                m_update    = beta1 * m + (1 - beta1) * (grad)
                v_update    = beta2 * v + (1 - beta2) * (grad ** 2)
                t_update    = t + 1

                adjusted_lr = lr * math.sqrt( 1 - (beta2 ** t) ) / (1 - beta1 ** t)

                p.data     -= adjusted_lr * m_update/(v_update.sqrt() + eps)
                p.data      = p.data - lr * wd * p.data

                state['m']  = m_update
                state['v']  = v_update
                state["t"]  = t_update                
                
                
        return loss
