from rwkvstic.agnostic.backends.torch import RWKVPTOps


class RWKVCoreMLOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, includeSampler=None, **kwargs):
        import torch
        super().__init__(layers, embed, dtype=torch.float32, *args, **kwargs)
        import torch
        import inquirer
        self.stack = lambda x: torch.stack(x)

        includeSampler = includeSampler

        if includeSampler:
            from rwkvstic.agnostic.samplers.torch import torchsample
            self.sample = torchsample
            self.postProcessTensor = lambda x: self.sample(
                x.float().cpu(), torch.tensor(1), torch.tensor(0.9))

        def exportTorchScript(x):
            traced_model = torch.jit.trace(
                x, (torch.LongTensor([0]), self.emptyState), check_trace=False, strict=False)
            self.emptyState = self.emptyState.float().cpu().numpy()

            import coremltools as ct

            # Using image_input in the inputs parameter:
            # Convert to Core ML program using the Unified Conversion API.
            model = ct.convert(
                traced_model,
                convert_to="mlprogram",
                inputs=[ct.TensorType(name="input", shape=(1,)),
                        ct.TensorType(name="state", shape=(5*self.n_layers, embed))],

            )

            class interop:
                def __init__(self, model):
                    self.model = model

                def __call__(self, x, state):
                    return self.model({"input": x, "state": state})

            return interop(model)

        self.postProcessModule = exportTorchScript
