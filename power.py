import torch


def powerTri(t, p):
    t = t.expand(p, p, -1)

    tri = ((torch.arange(p).expand(p,p)+1).t()-torch.arange(p)).tril().unsqueeze(-1)
    
    mask = torch.ones(p,p).tril().unsqueeze(-1)

    return t.pow(tri)*mask

mmm = 1

zm = powerTri(torch.tensor([1, 2, 3]), mmm)

zr = torch.tensor([[0.5, 0.2, 0.3]])
print(zm)

print(zr)

print((zr*zm).sum((1)))
