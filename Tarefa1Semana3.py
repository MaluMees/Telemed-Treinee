import torch

def funcao(tensor): 

    return torch.max(tensor, torch.zeros(tensor.size()))

x = torch.arange(-2.0, 2.0, 0.1).reshape(-1, 5)

print(x)
print(funcao(x))