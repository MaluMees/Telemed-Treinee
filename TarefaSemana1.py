import torch

def multiplicador(shape_a, shape_b):
    tensor_a = torch.randn(shape_a)
    tensor_b = torch.randn(shape_b)

    print(tensor_a)
    print(tensor_b)
    return torch.matmul(tensor_a, tensor_b)

x, y = tuple(map(int, input().split()))
w, z = tuple(map(int, input().split()))

shape1 = (x, y)
shape2 = (w, z)

print(multiplicador(shape1, shape2))