def typical(logits, temp=1.0, tau=0.95, **kwargs):
        import torch
    # do it in pytorch
        
        probs = torch.nn.functional.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
        shifted_logits = torch.abs(logits - ent)
        sorted_ids = torch.argsort(shifted_logits)
        sorted_logits = shifted_logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < tau)
        probs[shifted_logits > sorted_logits[cutoff]] = 0
        if temp != 1.0:
            probs = probs ** (1.0 / temp)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)