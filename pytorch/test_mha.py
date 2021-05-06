import torch

a = torch.randn((3, 2048, 7, 7))
b = torch.randn((2048,7,7))

# a = a.permute(2, 3, 0, 1)
# a = a.view(-1, 3, 2048)
# qry = a
# attn = torch.nn.MultiheadAttention(2048, num_heads=1, dropout=0.2, kdim=85, vdim=85)
# keys = torch.randn((1, 3, 85))
# values = torch.randn((1, 3, 85))
# # this is the class embeddings
# # att = torch.randn((85, 50))
#
# att = attn(qry, keys, values)

print()
