from rwkvstic.rwkvMaster import RWKVMaster
from rwkvstic.agnostic.samplers.numpy import npsample


def initRwkvRsFile(model_path, tokenizer=None):
    import rwkv_rs
    import huggingface_hub
    import os
    if not os.path.exists(model_path):
        model_path = huggingface_hub.hf_hub_download(
            repo_id="mrsteyk/RWKV-LM-safetensors", filename="RWKV-4-Pile-7B-Instruct-test1-20230124.rnn.safetensors")
    assert model_path is not None
    rsmodel = rwkv_rs.Rwkv(model_path)
    emptyState = rwkv_rs.State(rsmodel)

    class InterOp():
        def forward(self, x, y):
            logits = rsmodel.forward_token(x[-1], y)

            return logits, y
    model = InterOp()

    def initTensor(x): return x
    return RWKVMaster(model, emptyState, initTensor, npsample, tokenizer)
