import torch
import tqdm
import rwkvstic.tokenizer as tokenizer

# this is for like, being useful


def loadContext(model, ctx, newctx, statex, progressCallBack=lambda x: x):

    with torch.jit.optimized_execution(True):
        for i in tqdm.tqdm(range(len(newctx))):

            x = ctx+newctx[:i]

            o = model.forward([x[-1]], statex)
            statex = o[1]
            progressCallBack(x)
    return ctx+newctx, o[1]


class RWKVMaster():
    def __init__(self, model, emptyState, initTensor=lambda x: x, sampler=None):
        self.model = model
        self.tokenizer = tokenizer.tokenizer
        self.emptyState = emptyState
        self.myState = emptyState
        self.lastToken = 187
        self.initTensor = initTensor
        self.sampler = sampler

    def forward(self, state=None, temp: float = 1.0, top_p_usual: float = 0.8, number=1):
        state = self.myState if state is None else state
        tolens = []
        for i in range(number):
            logits, state = self.model.forward([self.lastToken], state)
            self.myState = state
            sampled = self.sample(
                logits, temp, top_p_usual) if self.sampler is not None else logits
            try:
                self.lastToken = sampled.item()
            except:
                self.lastToken = sampled

            tolens += [self.lastToken]
        sampled = self.tokenizer.decode(tolens)
        return {"logits": logits, "state": state, "output": sampled}

    def loadContext(self, ctx: str = "\n\n", newctx: str = "", statex=None, progressCallBack=lambda x: x):
        statex = self.myState if statex is None else statex
        ctx = self.tokenizer.encode(ctx)
        newctx = self.tokenizer.encode(newctx)
        ctx, state = loadContext(
            self.model, ctx, newctx, statex, progressCallBack)
        self.lastToken = ctx[-1]
        self.myState = state
        return ctx, state

    def sample(self, ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
        return self.sampler(ozut, temp, top_p_usual)

    def decode(self, x):
        return self.tokenizer.decode(x)

    def encode(self, x):
        return self.tokenizer.encode(x)

    def setState(self, state):
        self.myState = state

    def getState(self):
        return self.myState

    def resetState(self):
        self.myState = self.emptyState
