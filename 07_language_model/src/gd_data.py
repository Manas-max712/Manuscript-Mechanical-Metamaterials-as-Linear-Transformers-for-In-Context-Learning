from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

EOT = "<|endoftext|>"


def iter_eot_stories(lines):
    buffer = ""
    for line in lines:
        buffer += str(line).rstrip("\r\n") + "\n"
        while EOT in buffer:
            story, buffer = buffer.split(EOT, 1)
            if story.strip(): yield story.strip() + "\n" + EOT
    if buffer.strip(): yield buffer.strip() + "\n" + EOT


def canonical_text(text):
    return " ".join(text.replace(EOT, "").split())


def story_hash(text):
    return hashlib.sha256(canonical_text(text).encode()).hexdigest()


def token_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


class TokenDataset(Dataset):
    def __init__(self, root, split):
        self.ids = np.load(Path(root) / f"{split}_ids.npy", mmap_mode="r")
        self.lengths = np.load(Path(root) / f"{split}_lengths.npy", mmap_mode="r")
        if len(self.ids) != len(self.lengths): raise ValueError("Token/length mismatch")
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.ids[i], dtype=np.int64)), int(self.lengths[i])


def collate_tokens(examples):
    ids, lengths = zip(*examples)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return torch.stack(ids)[:, :int(lengths.max())], lengths


def input_targets(ids, lengths):
    x, y = ids[:, :-1], ids[:, 1:].clone()
    positions = torch.arange(y.shape[1], device=y.device)[None, :]
    y.masked_fill_(positions >= (lengths[:, None] - 1), -100)
    return x, y


class EpochBatches(Sampler):
    def __init__(self, size, batch_size, data_seed, epoch, start_batch=0):
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.start_batch = data_seed, epoch, start_batch
    def __len__(self):
        return max(0, (self.size+self.batch_size-1)//self.batch_size-self.start_batch)
    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + 1000003*self.epoch)
        order = torch.randperm(self.size, generator=g).tolist()
        for i in range(self.start_batch*self.batch_size, self.size, self.batch_size):
            yield order[i:i+self.batch_size]


def load_manifest(root):
    m = json.loads((Path(root)/"manifest.json").read_text())
    if m.get("schema") != "causal_language_revision_v1":
        raise ValueError("Use data produced by prepare_data.py")
    if "files" in m:
        from common import sha256, object_hash
        expected=object_hash({"settings":m["settings"],"files":m["files"]})
        if expected!=m["dataset_id"]:raise ValueError("Data manifest identity mismatch")
        for name,digest in m["files"].items():
            if sha256(Path(root)/name)!=digest:raise ValueError(f"Data checksum mismatch: {name}")
    return m
