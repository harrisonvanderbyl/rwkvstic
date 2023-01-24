def torchsample(ozut, temp=1.0, top_p_usual=0.8) -> int:
    import torch
    # do it in pytorch

    probs = torch.softmax(ozut, dim=-1)
    sorted_probs, indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = sorted_probs[torch.argmax(
        cumulative_probs[cumulative_probs > top_p_usual])]
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = torch.pow(probs, 1.0 / temp)
    probs = probs / torch.sum(probs, dim=-1)
    mout = torch.multinomial(probs, 1)
    return mout.cpu()
