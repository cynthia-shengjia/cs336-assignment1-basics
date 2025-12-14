from    einops import rearrange, einsum
import  torch

batch_size = 32
seq_len = 512
d_k = 64
d_v = 128

Q = torch.empty(seq_len,d_k)
K = torch.empty(seq_len,d_k)
V = torch.empty(seq_len,d_v)

attn_score = einsum(Q,K,"... i d, ... j d -> ... i j")

# x = torch.arange(1,11,1)

# seq_len     = 512
# batch_size  = 1024
# test_bach   = torch.stack([torch.stack([x for _ in range(512)], dim = 0) for _ in range(batch_size)],dim = 0)

# res = rearrange(test_bach, "... (x y) -> ... x y", y = 2)
# a = res[...,0]
# b = res[...,1]
# print(a[0,0,:])
# print(b[0,0,:])

# feature1 = torch.arange(start = 1, end = 10, step = 2)
# feature2 = torch.arange(start = 2, end = 11, step = 2)

# res = torch.stack([feature1,feature2], dim = 1)
# torch.flatten()

