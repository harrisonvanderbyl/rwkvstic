import torch
from rwkvstic.training.modules.block import Block




class RWKV(torch.nn.Module):

    def __init__(self,dims,layers, head, T):
        super(RWKV, self).__init__()
        print("Training RWKV")

        

        
        # head = 50277

        

        
        
        self.emb =  torch.nn.Embedding(head, dims)
        self.ln_out = torch.nn.LayerNorm((dims,))
        self.ln_in = torch.nn.LayerNorm((dims,))

        self.head = torch.nn.Linear(dims, head)

        self.register_module("emb", self.emb)
        self.register_module("ln_out", self.ln_out)
        self.register_module("ln_in", self.ln_in)
        self.register_module("head", self.head)

        # set all weights to 0
        self.emb.weight.data = torch.zeros_like(self.emb.weight.data)
        self.ln_out.weight.data = torch.zeros_like(self.ln_out.weight.data)
        self.ln_out.bias.data = torch.zeros_like(self.ln_out.bias.data)
        self.ln_in.weight.data = torch.zeros_like(self.ln_in.weight.data)
        self.ln_in.bias.data = torch.zeros_like(self.ln_in.bias.data)
        self.head.weight.data = torch.zeros_like(self.head.weight.data)
        self.head.bias.data = torch.zeros_like(self.head.bias.data)

        
        # loading bar
        from tqdm import tqdm
        self.blocks = torch.nn.ModuleList([Block(dims,T) for i in tqdm(range(layers), desc="loading layers")])

    def forward(self, x,):
        
        x = self.emb(x)
        x = self.ln_in(x)

        for i, block in enumerate(self.blocks):

            x = block(x)

        x = self.ln_out(x)

        outx = self.head(x)
        

        return outx
    

    

dims = 256
layers = 20
from rwkvstic.tokenizer import tokenizer
t = tokenizer()
myrwkv = RWKV(dims,layers, head = 140, T=100)

# open text file
with open("/home/harrison/Desktop/rwkvstic/src/rwkvstic/training/data/t8.shakespeare.txt", "r") as f:
    text = f.read()

# get ascii values
data = [ord(i) for i in text]
print (data[:100])
print(max(data))

myrwkv.to("cuda")
myrwkv.train()
for x in myrwkv.parameters():
    x.requires_grad = True

for epoch in range(len(data)//100):
    
    out = myrwkv.forward(torch.LongTensor(data[epoch*100:epoch*100+100]).to(torch.int64).to("cuda"))
    print(out)

    loss = torch.nn.CrossEntropyLoss()
    m = loss(out, (torch.LongTensor(data[epoch*100+1:epoch*100+101]).to(torch.int64).to("cuda")))
    print(m)
    m.backward()
    
    

    # update parameters
    optimizer = torch.optim.Adam(myrwkv.parameters(), lr=0.1/(1+epoch/100))
    optimizer.step()





