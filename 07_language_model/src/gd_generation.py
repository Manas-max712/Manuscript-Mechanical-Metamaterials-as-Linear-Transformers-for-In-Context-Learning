import torch
import torch.nn.functional as F


def banned_ngram(ids, n):
    if n<1 or len(ids)<n-1:return set()
    prefix=tuple(ids[-(n-1):]) if n>1 else ()
    return {ids[i+n-1] for i in range(len(ids)-n+1)
            if tuple(ids[i:i+n-1])==prefix}


@torch.no_grad()
def next_logits(model, ids, device, pad_id=None):
    if not ids:raise ValueError("Provide a nonempty prompt")
    context=ids[-model.n_ctx:]
    logits=model(torch.tensor([context],dtype=torch.long,device=device))[0,-1].clone()
    if pad_id is not None:logits[pad_id]=-torch.inf
    return logits


@torch.no_grad()
def beam_generate(model, prefix, device, max_new_tokens=50, num_beams=3,
                  eos_id=None, pad_id=None, no_repeat_ngram=3,length_penalty=1.0):
    if num_beams<1 or max_new_tokens<1:raise ValueError("Positive beam width/length required")
    model.eval(); prefix=list(map(int,prefix))
    beams=[dict(new=[],score=0.0,finished=False)]
    def rank(b):return b["score"]/max(len(b["new"]),1)**length_penalty
    for _ in range(max_new_tokens):
        candidates=[]
        for b in beams:
            if b["finished"]:candidates.append(b);continue
            history=prefix+b["new"]
            logp=F.log_softmax(next_logits(model,history,device,pad_id),dim=-1)
            blocked=banned_ngram(history,no_repeat_ngram)
            if blocked:logp[list(blocked)]=-torch.inf
            scores,ids=torch.topk(logp,k=min(num_beams,logp.numel()))
            for score,token in zip(scores.tolist(),ids.tolist()):
                if not torch.isfinite(torch.tensor(score)):continue
                candidates.append(dict(new=b["new"]+[token],score=b["score"]+score,
                                       finished=(eos_id is not None and token==eos_id)))
        if not candidates:raise RuntimeError("No finite beam candidate")
        beams=sorted(candidates,key=rank,reverse=True)[:num_beams]
        if all(b["finished"] for b in beams):break
    best=max(beams,key=rank)
    return best


@torch.no_grad()
def sample_generate(model,prefix,device,max_new_tokens=50,eos_id=None,pad_id=None,
                    temperature=.8,top_k=40,no_repeat_ngram=3,generator=None):
    if temperature<=0:raise ValueError("temperature must be positive")
    model.eval();ids=list(map(int,prefix));start=len(ids)
    for _ in range(max_new_tokens):
        logits=next_logits(model,ids,device,pad_id)/temperature
        blocked=banned_ngram(ids,no_repeat_ngram)
        if blocked:logits[list(blocked)]=-torch.inf
        values,indices=torch.topk(logits,k=min(top_k,logits.numel()))
        if not torch.isfinite(values).any():raise RuntimeError("No finite sampling candidate")
        token=int(indices[torch.multinomial(values.softmax(0),1,generator=generator)].item())
        ids.append(token)
        if eos_id is not None and token==eos_id:break
    return ids[start:]
