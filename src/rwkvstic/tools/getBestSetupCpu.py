import time
import torch

mat = torch.randn(1000, 1000).float()
vec = torch.randn(1000).float()
matbfloat = mat.bfloat16()
vecbfloat = vec.bfloat16()

matdouble = mat.double()
vecdouble = vec.double()

rounds = 1000

# warmup
for i in range(1000):
    x = torch.mv(mat, vec)


# time the bfloat16 matmul


start = time.time()
for i in range(rounds):
    x = torch.mv(matbfloat, vecbfloat)

end = time.time()


# time the float matmul

start = time.time()
for i in range(rounds):
    x = torch.mv(mat, vec)

end = time.time()


start = time.time()
for i in range(rounds):
    x = torch.mv(matbfloat, vecbfloat)

end = time.time()

print(f"bfloat16 matmul: {end-start} ms")


# time the float matmul

start = time.time()
for i in range(rounds):
    x = torch.mv(mat, vec)

end = time.time()

print(f"float matmul: {end-start} ms")
