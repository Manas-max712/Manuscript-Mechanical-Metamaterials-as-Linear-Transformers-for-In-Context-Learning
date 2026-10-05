import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gd_data import collate_tokens, input_targets


def loss_sum(logits, targets):
    count = int(targets.ne(-100).sum())
    if not count: raise ValueError("Batch has no valid next-token targets")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100, reduction="sum"), count


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval(); total, ntokens = 0.0, 0
    position_sum = torch.zeros(model.n_ctx, dtype=torch.float64)
    position_count = torch.zeros(model.n_ctx, dtype=torch.int64)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_tokens,
                        shuffle=False, generator=torch.Generator().manual_seed(0))
    for ids, lengths in loader:
        ids, lengths = ids.to(device), lengths.to(device)
        x, y = input_targets(ids, lengths); logits = model(x)
        losses = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
        valid = y.ne(-100)
        if not torch.isfinite(losses).all(): raise FloatingPointError("Nonfinite evaluation loss")
        total += float(losses.double().sum()); ntokens += int(valid.sum())
        S = y.shape[1]
        position_sum[:S] += losses.double().sum(0).cpu()
        position_count[:S] += valid.sum(0).cpu()
    if not ntokens: raise ValueError("Evaluation set is empty")
    ce = total/ntokens
    return dict(ce=ce, perplexity=math.exp(ce) if ce < 700 else float("inf"),
                token_count=ntokens, loss_sum=total,
                position_ce=[float(s/n) if n else None for s,n in zip(position_sum,position_count)],
                position_tokens=position_count.tolist())
