import torch
from common import make_model

@torch.no_grad()
def check_causality(model,ids,atol=1e-8,rtol=1e-9):
    model.eval();ids=ids[:,:min(12,model.n_ctx)]
    n=ids.shape[1];pivot=max(1,n//2)
    info=dict(device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype),
              atol=atol,rtol=rtol,input_tokens=n,tested_positions=pivot)
    if n<2:return dict(**info,passed=False,status="insufficient_input")
    other=ids.clone();other[:,pivot:]=(other[:,pivot:]+17)%model.wte.num_embeddings
    a=model(ids);b=model(other);p=model(ids[:,:pivot])
    future=float((a[:,:pivot]-b[:,:pivot]).abs().max())
    prefix=float((p-a[:,:pivot]).abs().max())
    finite=all(bool(torch.isfinite(z).all()) for z in [a,b,p])
    passed=finite and torch.allclose(a[:,:pivot],b[:,:pivot],atol=atol,rtol=rtol) and torch.allclose(p,a[:,:pivot],atol=atol,rtol=rtol)
    return dict(**info,future_logit_max_abs=future if finite else None,
                prefix_logit_max_abs=prefix if finite else None,finite_logits=finite,
                passed=passed,status="passed" if passed else "failed")

def cpu_checkpoint_causality(cfg,state,ids):
    cpu_model=make_model(cfg,torch.device("cpu")).double()
    cpu_model.load_state_dict(state,strict=True)
    return check_causality(cpu_model,ids.cpu())
