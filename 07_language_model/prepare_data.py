#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent / "src"))
import argparse
import fcntl
import hashlib
import heapq
import json
import os
import time
import numpy as np
from gd_data import EOT, iter_eot_stories, canonical_text, story_hash, token_hash
from common import SCHEMA, atomic_json, object_hash, sha256


def collect_sample(records, capacity, seed):
    seen, heap, counts = set(), [], dict(read=0, duplicate_stories=0, empty_stories=0)
    for text, source in records:
        counts["read"] += 1
        canonical = canonical_text(text)
        if not canonical:
            counts["empty_stories"] += 1; continue
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest in seen:
            counts["duplicate_stories"] += 1; continue
        seen.add(digest)
        priority = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest(), 16)
        item = (-priority, digest, text.replace(EOT, "").strip() + "\n" + EOT, source)
        if len(heap) < capacity: heapq.heappush(heap, item)
        elif item[0] > heap[0][0]: heapq.heapreplace(heap, item)
        if counts["read"] % 250000 == 0:
            print(f"Scanned {counts['read']:,} stories; retained reservoir {len(heap):,}", flush=True)
    counts["unique_stories"] = len(seen)
    return sorted(heap, key=lambda x: (-x[0], x[1])), counts


def remote_records(provenance):
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    api = HfApi()
    tiny_repo = "roneneldan/TinyStories"
    tiny_revision = api.dataset_info(tiny_repo).sha
    provenance["TinyStories"] = {"repo": tiny_repo, "revision": tiny_revision, "files": {}}
    for name in ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]:
        path = hf_hub_download(tiny_repo, name, repo_type="dataset", revision=tiny_revision)
        provenance["TinyStories"]["files"][name] = sha256(path)
        with open(path, encoding="utf-8") as lines:
            for story in iter_eot_stories(lines): yield story, "TinyStories"
    repo = "ajibawa-2023/Children-Stories-Collection"
    revision = api.dataset_info(repo).sha
    provenance["ChildrenStories"] = {"repo": repo, "revision": revision}
    dataset = load_dataset(repo, revision=revision)
    for split in sorted(dataset.keys()):
        for row in dataset[split]:
            text = row.get("text")
            if isinstance(text, str) and text.strip(): yield text, "ChildrenStories"


def local_records(files, provenance):
    provenance["local_eot_files"] = {str(Path(p).resolve()): sha256(p) for p in files}
    for filename in files:
        with open(filename, encoding="utf-8") as lines:
            for story in iter_eot_stories(lines): yield story, Path(filename).name


def prepare(args):
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(n_ctx=args.n_ctx, train_stories=args.train_stories,
                    valid_stories=args.valid_stories, test_stories=args.test_stories,
                    data_seed=args.data_seed, local_eot=args.local_eot,
                    tokenizer=args.tokenizer, schema=SCHEMA)
    with output.with_name(output.name + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            manifest_file = output / "manifest.json"
            if manifest_file.exists():
                saved = json.loads(manifest_file.read_text())
                if saved["settings"] != settings:
                    raise RuntimeError("Prepared-data settings differ; use a new output directory")
                for name, digest in saved["files"].items():
                    if sha256(output/name) != digest: raise RuntimeError(f"Data checksum mismatch: {name}")
                print(f"Verified prepared data: {output}"); return saved
            raise RuntimeError("Output directory exists without manifest; use a new directory")
        from transformers import GPT2TokenizerFast
        from huggingface_hub import HfApi
        t0 = time.monotonic(); provenance = {}
        tokenizer_revision = None
        if not Path(args.tokenizer).is_dir():
            tokenizer_revision = HfApi().model_info(args.tokenizer).sha
        tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer, revision=tokenizer_revision)
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"}); tokenizer.padding_side = "right"
        provenance["tokenizer"] = dict(name=args.tokenizer, revision=tokenizer_revision)
        raw = local_records(args.local_eot, provenance) if args.local_eot else remote_records(provenance)
        requested = args.train_stories+args.valid_stories+args.test_stories
        selected, counts = collect_sample(raw, requested+max(1000, requested//20), args.data_seed)
        staging = output.with_name(output.name+f".building-{os.getpid()}")
        staging.mkdir(exist_ok=False); tokenizer.save_pretrained(staging/"tokenizer")
        quotas = {"valid": args.valid_stories, "test": args.test_stories, "train": args.train_stories}
        arrays, lengths, hashes, streams = {}, {}, {}, {}
        for split, n in quotas.items():
            arrays[split] = np.lib.format.open_memmap(staging/f"{split}_ids.npy", mode="w+",
                                                     dtype=np.int32, shape=(n,args.n_ctx+1))
            arrays[split][:] = tokenizer.pad_token_id
            lengths[split] = np.lib.format.open_memmap(staging/f"{split}_lengths.npy", mode="w+",
                                                      dtype=np.int16, shape=(n,))
            hashes[split] = []
            if split != "train": streams[split] = (staging/f"{split}_stories.jsonl").open("w")
        completed = {s:0 for s in quotas}; token_seen=set(); skipped_window=0; skipped_short=0
        mix = {s:{} for s in quotas}
        try:
            for offset in range(0,len(selected),1024):
                entries=selected[offset:offset+1024]
                encoded=tokenizer([x[2] for x in entries],truncation=True,max_length=args.n_ctx+1,
                                  add_special_tokens=False)["input_ids"]
                for entry, ids in zip(entries, encoded):
                    if len(ids)<2: skipped_short+=1; continue
                    digest=token_hash(ids)
                    if digest in token_seen: skipped_window+=1; continue
                    token_seen.add(digest)
                    split=next((s for s in quotas if completed[s]<quotas[s]),None)
                    if split is None: break
                    i=completed[split]; arrays[split][i,:len(ids)]=ids; lengths[split][i]=len(ids)
                    hashes[split].append(entry[1]); completed[split]+=1
                    mix[split][entry[3]]=mix[split].get(entry[3],0)+1
                    if split in streams:
                        streams[split].write(json.dumps(dict(story_sha256=entry[1],source=entry[3],text=entry[2]))+"\n")
                if sum(completed.values())==requested: break
        finally:
            for stream in streams.values():stream.close()
        if completed != quotas:
            raise RuntimeError(f"Insufficient distinct token windows: {completed}, requested {quotas}. Staging retained at {staging}")
        for split in quotas:
            arrays[split].flush(); lengths[split].flush()
            (staging/f"{split}_story_hashes.txt").write_text("\n".join(hashes[split])+"\n")
        files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest=dict(schema=SCHEMA,settings=settings,sources=provenance,files=files,
                      dataset_id=object_hash({"settings":settings,"files":files}), counts=completed,
                      source_counts=mix, parser_counts=counts, duplicate_token_windows_skipped=skipped_window,
                      short_stories_skipped=skipped_short,pad_id=tokenizer.pad_token_id,
                      eos_id=tokenizer.eos_token_id,vocab_size=len(tokenizer),n_ctx=args.n_ctx,
                      max_sequence_tokens=args.n_ctx+1,build_seconds=time.monotonic()-t0,
                      split_policy="Deduplicate full normalized stories; select by seeded SHA256 priority; valid, test, train allocation; deduplicate retained token windows across all splits.")
        atomic_json(staging/"manifest.json",manifest); os.replace(staging,output)
        print(json.dumps({k:manifest[k] for k in ["dataset_id","counts","build_seconds"]},indent=2))
        return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True);p.add_argument("--n-ctx",type=int,default=64)
    p.add_argument("--train-stories",type=int,default=300000)
    p.add_argument("--valid-stories",type=int,default=2000)
    p.add_argument("--test-stories",type=int,default=2000)
    p.add_argument("--data-seed",type=int,default=0)
    p.add_argument("--tokenizer",default="openai-community/gpt2")
    p.add_argument("--local-eot",nargs="+",default=None,help="Explicit local raw files for offline tests/data preparation")
    args=p.parse_args()
    if min(args.train_stories,args.valid_stories,args.test_stories,args.n_ctx)<1:p.error("Counts must be positive")
    prepare(args)


if __name__=="__main__":main()
