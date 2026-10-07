import torch

x = torch.arange(12, dtype = torch.float32) #define o tensor
print(x)
print(x.numel()) #metodo q da numero de elementos no tensor
print(x.shape) #atributo de forma q da numero de elementos em cada eixo
x = x.reshape(3, 4) #muda a forma do tensor para uma matriz 3x4
#For example, given a tensor of size "n" and target shape (h,w), we know that w = n/h. To automatically infer one component of the shape, we can place a -1 for the shape component that should be inferred automatically. In our case, instead of calling x.reshape(3, 4), we could have equivalently called x.reshape(-1, 4) or x.reshape(3, -1).
print(x)

y = torch.zeros((2, 3, 4))
print(y)
z = torch.ones((2, 3, 4))
print(z)
i = torch.randn(3, 4) #cria um tensor com valores aleatórios
t = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])#assim tu constroi escolhendo os valores no tensor
print(t)
print(t[-1])#corta o tensor pegando a ultima linha
print(t[1:3])#corta o tensor pegando a segunda e terceira linha
t[1, 2] = 17#muda o elemento em um lugar específico da matriz
print(t)
t[:2, :] = 12 #tranforma os elementos das duas primeira linhas em 12. Funciona para vetores com mais de duas dimensões tbm.
print(t)
v = torch.exp(x) #
print(v)
w = torch.arange(12, dtype=torch.float32).reshape((3,4))
s = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
q = torch.cat((w, s), dim=0) #concatena as linhas verticalmente
d = torch.cat((w, s), dim=1) #concatena as linhas horizontalmente
print(w)
print(s)
print(d)