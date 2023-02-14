# RWKVSTIC

Rwkvstic, pronounced however you want to, is a library for interfacing and using the RWKV-V4 based models.

Rwkvstic does not autoinstall its dependencies, as its main purpose is to be dependency agnostic, able to be used by whatever library you would prefer.

When using BlinkDLs pretrained models, it would advised to have the `torch` package installed.

Some options, when left blank, will elicit a prompt asking you to choose a value.
for this purpose, please ensure you have the `inquirer` package installed.

## Tables and graphs

### Rwkv-4 models -> recomended vram

```
rwkvstic vram
Model | 8bit | bf16/fp16 | fp32
14B   | 16GB | 28GB      | >50GB
7B    | 8GB  | 14GB      | 28GB
3B    | 2.8GB| 6GB       | 12GB
1b5   | 1.3GB| 3GB       | 6GB
```

## Installation

```bash
pip install rwkvstic
```

## Basic Usage

```python
from rwkvstic.load import RWKV

# Load the model (supports full path, relative path, and remote paths)

model = RWKV(
    "https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-Instruct-test1-20230124.pth"
)

model.loadContext(newctx=f"Q: who is Jim Butcher?\n\nA:")
output = model.forward(number=100)["output"]

print(output)

# Q: who is Jim Butcher?
# A: Jim Butcher is a very popular American author of fantasy novels. He’s known for the Dresden Files series of novels.<|endoftext|>
```

## Advanced Usage

#

## Step 1: load the model with your choice of poison

### Pytorch

```python
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH

# this is the dtype used for trivial operations, such as vector->vector operations and is the dtype that will determine the accuracy of the model
runtimedtype = torch.float32 # torch.float64, torch.bfloat16

# this is the dtype used for matrix-vector operations, and is the dtype that will determine the performance and memory usage of the model
dtype = torch.bfloat16 # torch.float32, torch.float64, torch.bfloat16

useGPU = True # False

model = RWKV("path/to/model.pth", backend=TORCH, useGPU=useGPU, runtimedtype=runtimedtype, dtype=dtype)
```

### JAX

```python
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import JAX

# Jax will automatically use the GPU if available, and will use the CPU if not available

model = RWKV("path/to/model.pth", backend=JAX)
```

### TensorFlow

```python
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TF

useGPU = True # False

model = RWKV("path/to/model.pth", backend=TF, useGPU=useGPU)
```

### Numpy

```python
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import NUMPY

# you masochistic bastard
model = RWKV("path/to/model.pth", backend=NUMPY)
```

### Streaming

#### Trade vram usage for performance

```python
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH_STREAM

# this is the dtype used for trivial operations, such as vector->vector operations and is the dtype that will determine the accuracy of the model
runtime_dtype = torch.float32 # torch.float64, torch.bfloat16

# this is the dtype used for matrix-vector operations, and is the dtype that will determine the performance and memory usage of the model
dtype = torch.bfloat16 # torch.float32, torch.float64, torch.bfloat16

# this is the amount of GB you want to use for matrix storage, if the model is too large, matrixes will be stored in ram and moved to the GPU as needed
target = 4

# Pin Memory is used to speed up the transfer of data to the GPU, but will use more memory, both on the GPU and on the CPU
pin_memory = True

model = RWKV("path/to/model.pth", backend=TORCH_STREAM, runtimedtype=runtime_dtype, dtype=dtype, target=target, pinMem=pin_memory)

```

### Multi-GPU

#### Model weights are split(sharded) across multiple GPUs

```python
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH_SPLIT

# this is the dtype used for trivial operations, such as vector->vector operations and is the dtype that will determine the accuracy of the model
runtime_dtype = torch.float32 # torch.float64, torch.bfloat16

# this is the dtype used for matrix-vector operations, and is the dtype that will determine the performance and memory usage of the model
dtype = torch.bfloat16 # torch.float32, torch.float64, torch.bfloat16

model = RWKV("path/to/model.pth", backend=TORCH_SPLIT, runtimedtype=runtime_dtype, dtype=dtype)

```

### Quantization

#### Uses close to half the memory of float16, but is slightly less accurate, and is about 4x slower

```python
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH_QUANT

# this is the dtype used for trivial operations, such as vector->vector operations and is the dtype that will determine the accuracy of the model
runtime_dtype = torch.float32 # torch.float64, torch.bfloat16

# this is the amount of chunks to split the matrix rows into pre-row-quantization, the more chunks, the more accurate the model will be, but with some minor trade offs
chunksize = 4

useGPU = True # False

# this is the amount of GB you want to use for matrix storage, if the model is too large, matrixes will be stored in ram and moved to the GPU as needed, same as stream
target = 4

model = RWKV("path/to/model.pth", backend=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU, target=target)
```

## Step 2: State management

### The state

The state is a vectorized value that is a representation of all the previous inputs and outputs of the model. It is used basically the memory of the model, and is used to generate the next output.

The model has an internal state, so the following is useful in that regards.

```python
model = RWKV("path/to/model.pth")

emptyState = model.emptyState()
model.setState(emptyState)
currentMem = model.getState()
```

## Step 3: Injecting context

### Injecting context

When you want to influence the output of the model, you can inject context into the model. This is done by using the `loadContext` function.

```python
model = RWKV("path/to/model.pth")

model.loadContext(newctx="Q: who is Jim Butcher?\n\nA:")

print(model.forward(number=100)["output"])

model.loadContext(newctx="Can you tell me more?\n\nA:")
```

## Step 4: Generating output

### Generating output

When you want to generate output, you can use the `forward` function.

```python
model = RWKV("path/to/model.pth")

number = 100 # the number of tokens to generate
stopStrings = ["\n\n"] # When read, the model will stop generating output

stopTokens = [0] # advanced, when the model has generated any of these tokens, it will stop generating output

temp = 1 # the temperature of the model, higher values will result in more random output, lower values will result in more predictable output

top_p = 0.9 # the top_p of the model, higher values will result in more random output, lower values will result in more predictable output

def progressLambda(properties):
    # "logits", "state", "output", "progress", "tokens", "total", "current"
    print("progress:",properties["progress"]/properties["total"])

output = model.forward(number=number, stopStrings=stopStrings, stopTokens=stopTokens, temp=temp, top_p=top_p, progressLambda=progressLambda)

print(output["output"]) # the generated output
print(output["state"]) # the state of the model after generation
print(output["logits"]) # the logits of the model after generation, before sampling
```

# Implementation Details

## The RWKVOP object

Here is a base class, when overwritten, will allow the swapout of operations with their equivilents in different frameworks. Ill show you the JAX one, as its relatively simple

```python

class RWKVJaxOps(RWKVOp.module):
    def __init__(self, layers, embed, preJax=False):
        from jax import numpy as npjax
        super().__init__(layers, embed)
        # convert from torch to jax
        self.initTensor = lambda x: npjax.array(x.float().cpu().numpy())
        # jax math functions
        self.sqrt = lambda x: npjax.sqrt(x)
        self.mean = lambda x: npjax.mean(x)
        self.relu = lambda x: npjax.maximum(x, 0)
        self.exp = lambda x: npjax.exp(x)
        self.matvec = npjax.matmul
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: npjax.minimum(x, y)
        self.log = npjax.log
        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln

        # constants and stuff
        self.klimit = npjax.array([18] * embed)
        self.stack = lambda x: x

        # module def
        self.module = object

        # function overwrites (used for advanced stuff)
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        # The empty state
        self.emptyState = npjax.array([[0.01]*embed]*4*layers)
```

This can then be used to construct and infer the model.

## Stream, Split And Quant

The stream, split and quant backends are all pytorch varients that use some tricks to use less, or distribute memory usage across multiple GPUs.

Ill show you the important stuff, usually consisting of how the matrixes are constructed, and how they are used to create a matvec.

(Disclaimer, just similar to the actual code, not the actual code, actual code is messy and gross)

### Stream

```python
# Pinning memory allows for faster transfer between CPU and GPU, but uses more memory
def pinmem(x):
            return x.pin_memory() if pinMem and x.device == "cpu" else x


def initMatrix(x):
    # if more memory is used then the target specified, then it is sent to the cpu
    if torch.cuda.max_memory_reserved(0)/1024/1024/1024 > target:
        x = x.cpu()
    else:
        x = x.cuda(non_blocking=True)
    return pinmem(x)

# for the matvec, it just brings it to the correct device as needed
def matvec(z, y):
    return z.to(y.device, non_blocking=True) @ y
```

### Split

```python
def initMatrix(x):
    devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    # split the matrix into the number of devices
    x = torch.split(x, x.shape[0]//len(devices), dim=0)
    # send each part to a different device
    x = [i.to(devices[i], non_blocking=True) for i in range(len(x))]
    return x

# for the matvec, split the vector into the number of devices, and then send each part to the correct device
def matvec(z, y):
    devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    y = torch.split(y, y.shape[0]//len(devices), dim=0)
    y = [i.to(devices[i], non_blocking=True) for i in range(len(y))]
    # do the matvec on each part
    z = [z[i].mv(y[i]) for i in range(len(z))]
    # put them all on one device
    z = [i.to(devices[0], non_blocking=True) for i in z]
    # add them all together
    z = torch.sum(torch.stack(z), dim=0)
    return z
```

### Quant

```python
def QuantizeMatrix(x, runtimeDtype, device):
    rang = 255
    ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
    x = x.double()
    x = ((x-mini)/ran)

    x = x.to(
        dtype=torch.uint8, non_blocking=True, device=device)

    return x, ran.to(runtimeDtype).to(device=device), mini.to(runtimeDtype).to(device=device)

def MatVec(x, y, runtimedtype):
    # resize y into a 2d array
    y = y.reshape(chunksize, -1)

    # retrieve the  quantized matrix, the spread, and the offset
    rx, spread, zpoint = x

    # spread the y vector across the spread matrix
    yy = y*spread

    # convert the quantized matrix back to the runtime dtype
    rx = rx.to(dtype=runtimedtype)

    # we can use matmul to do a batched matvec for each split matrix
    xmain = rx.matmul(yy.reshape(yy.shape[0], -1, 1)).sum(0).squeeze()

    # the offset is added to the result
    return xmain + torch.tensordot(zpoint, y)

def initMatrix(x):
    # by splitting the matrix before quantizing, it allows for much better results
    splitmatrices = torch.chunk(x, chunksize, 1)
    xx = [QuantizeMatrix(x, runtimedtype, dev)
            for x in splitmatrices]
    xxo = torch.stack([x[0] for x in xx])
    xx1 = torch.stack([x[1] for x in xx])
    xx2 = torch.stack([x[2] for x in xx])
    return xxo, xx1, xx2
```

## PreQuantization

You can prequantize the matrixes to save loading time, and bandwidth when downloading model.

```bash
cd /path/to/folder/with/model
python3 -m rwkvstic --pq

# what model to prequantize?
# -> model.pth

ls
# model.pth
# model.pqth
```

You can load these pre-quantized models as you would a normal file.

```python
from rwkvstic.load import RWKV

model = RWKV("model.pqth")
```
