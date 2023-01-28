# RWKVSTIC

Rwkvstic, pronounced however you want to, is a library for interfacing and using the RWKV-V4 based models.

Rwkvstic does not autoinstall its dependencies, as its main purpose is to be dependency agnostic, able to be used by whatever library you would prefer.

When using BlinkDLs pretrained models, it would advised to have the `torch` package installed.

Some options, when left blank, will elicit a prompt asking you to choose a value.
for this purpose, please ensure you have the `inquirer` package installed.

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

model = RWKV("path/to/model.pth", backend=TORCH, useGPU=useGPU, runtimedtype=runtimdtype, dtype=dtype)
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

model = RWKV("path/to/model.pth", backend=TORCH_QUANT, runtimedtype=runtime_dtype, chunksize=chunksize, useGPU=useGPU)
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
