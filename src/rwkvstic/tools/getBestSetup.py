import torch

mat = torch.randn(1000, 1000).float().cuda()
vec = torch.randn(1000).float().cuda()
matbfloat = mat.bfloat16().cuda()
vecbfloat = vec.bfloat16().cuda()
mathalf = mat.half().cuda()
vechalf = vec.half().cuda()

rounds = 1000

# warmup
for i in range(1000):
    x = torch.mv(mat, vec)


# time the bfloat16 matmul
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for i in range(rounds):
    x = torch.mv(matbfloat, vecbfloat)

end.record()
torch.cuda.synchronize()


# time the half matmul
start.record()
for i in range(rounds):
    x = torch.mv(mathalf, vechalf)

end.record()
torch.cuda.synchronize()

# time the float matmul

start.record()
for i in range(rounds):
    x = torch.mv(mat, vec)

end.record()
torch.cuda.synchronize()

start.record()
for i in range(rounds):
    x = torch.mv(matbfloat, vecbfloat)

end.record()
torch.cuda.synchronize()
print(f"bfloat16 matmul: {start.elapsed_time(end)} ms")

# time the half matmul
start.record()
for i in range(rounds):
    x = torch.mv(mathalf, vechalf)

end.record()
torch.cuda.synchronize()
print(f"half matmul: {start.elapsed_time(end)} ms")

# time the float matmul

start.record()
for i in range(rounds):
    x = torch.mv(mat, vec)

end.record()
torch.cuda.synchronize()
print(f"float matmul: {start.elapsed_time(end)} ms")
