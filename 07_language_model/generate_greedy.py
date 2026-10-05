#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

PAPER_PROMPTS = [
    'Once upon a time, there was a quiet little dog named Spot. Spot loved to play with his',
    'In a tiny world full of wonders, lived minuscule creatures called microbes. They were too small to be seen by our eyes, but they had',
    'Once upon a time, in a small village named Harmonyville, lived two best friends, Sam and Alex. They loved exploring old buildings',
]
PREVIOUS_PROMPTS = [
    'Once upon a time, there was a little girl named Lily who found a shiny key in the garden near her house.',
    'Tom had a little dog named Max. One sunny morning, Max ran into the deep forest and Tom followed.',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--run', type=Path, default=Path('runs/causal_prefix_seed0'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gen-tokens', type=int, default=10)
    p.add_argument('--validation-prompts', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--require-cuda', action='store_true')
    a = p.parse_args()
    if a.gen_tokens < 1 or a.validation_prompts < 0 or a.cpu_threads < 1:
        p.error('Invalid count')
    package = a.package_dir.resolve()
    sys.path.insert(0, str(package / 'src'))
    import torch
    from transformers import GPT2TokenizerFast
    from gd_data import load_manifest
    from gd_generation import beam_generate, sample_generate
    from common import SCHEMA, make_model, seed_all, sha256, code_hashes, atomic_json

    def absolute(path):
        return (path if path.is_absolute() else package / path).resolve()

    run, data, output = map(absolute, (a.run, a.data, a.output))
    if output == package / 'src' or package / 'src' in output.parents:
        raise ValueError('Do not write inside src')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.require_cuda and device.type != 'cuda':
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(a.cpu_threads)
    seed_all(0)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    metadata = load_manifest(data)
    checkpoint = run / 'checkpoints/best.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('schema') != SCHEMA or ckpt.get('dataset_id') != metadata['dataset_id']:
        raise RuntimeError('Checkpoint and dataset do not match')
    model = make_model(ckpt['model_config'], device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    for name, value in model.state_dict().items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite model tensor {name}')
    def check_logits(_model, _args, logits):
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite model logits during generation')
    model.register_forward_hook(check_logits)
    tok = GPT2TokenizerFast.from_pretrained(data / 'tokenizer', local_files_only=True)
    prompts = [dict(kind='supplement_Table_S14', prompt=s) for s in PAPER_PROMPTS]
    prompts += [dict(kind='additional_fixed_prompt', prompt=s) for s in PREVIOUS_PROMPTS]
    for r in prompts:
        r['prefix_token_ids'] = tok.encode(r['prompt'], add_special_tokens=False)
    if a.validation_prompts:
        with (data / 'valid_stories.jsonl').open() as f:
            for i, line in enumerate(f):
                if i >= a.validation_prompts:
                    break
                story = json.loads(line)
                ids = tok.encode(story['text'], add_special_tokens=False)[:model.n_ctx + 1]
                half = max(1, len(ids) // 2)
                prompts.append(dict(kind='validation_completion', story_sha256=story['story_sha256'],
                    prompt=tok.decode(ids[:half]), prefix_token_ids=ids[:half],
                    reference_continuation=tok.decode(ids[half:half + a.gen_tokens], skip_special_tokens=True)))
        if len(prompts) != 5 + a.validation_prompts:
            raise RuntimeError('Insufficient validation stories')
    if any(not r['prefix_token_ids'] for r in prompts):
        raise RuntimeError('An empty prompt cannot be decoded')
    output.mkdir(parents=True, exist_ok=False)
    settings = [dict(decoder='greedy', no_repeat_ngram=0)]
    result = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  epoch=ckpt['epoch'], global_step=ckpt['global_step'], val_ce=ckpt.get('val_ce'),
                  dataset_id=metadata['dataset_id'], model_config=ckpt['model_config'],
                  torch_version=torch.__version__, device=str(device),
                  gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                  source_sha256=code_hashes(), script_sha256=sha256(__file__),
                  max_new_tokens=a.gen_tokens, temperature=.8, top_k=40, length_penalty=1.,
                  causality=dict(status='not_run', passed=None), settings=settings, records=[])
    start = time.monotonic()

    def save():
        result['elapsed_seconds'] = time.monotonic() - start
        atomic_json(output / 'generations.json', result)
        chunks = []
        for i, prompt in enumerate(prompts):
            chunks.append(f"PROMPT {i+1} [{prompt['kind']}]: {prompt['prompt']}")
            if 'reference_continuation' in prompt:
                chunks.append('REFERENCE: ' + prompt['reference_continuation'])
            for record in result['records']:
                if record['prompt_index'] == i:
                    chunks.append(f"{record['decoder']} / no_repeat_ngram={record['no_repeat_ngram']}: "
                                  + record.get('generated', 'ERROR: ' + record.get('error', '')))
            chunks.append('')
        temporary = output / 'generations.txt.tmp'
        temporary.write_text('\n'.join(chunks) + '\n')
        temporary.replace(output / 'generations.txt')

    save()
    for i, prompt in enumerate(prompts):
        for setting in settings:
            record = dict(**prompt, **setting, prompt_index=i, seed=1000 + i)
            print(f"Prompt {i+1}/{len(prompts)} {setting}", flush=True)
            try:
                options = dict(max_new_tokens=a.gen_tokens, eos_id=metadata['eos_id'],
                               pad_id=metadata['pad_id'], no_repeat_ngram=setting['no_repeat_ngram'])
                if setting['decoder'] == 'sample':
                    generator = torch.Generator(device=device).manual_seed(1000 + i)
                    ids = sample_generate(model, prompt['prefix_token_ids'], device, **options,
                                          temperature=.8, top_k=40, generator=generator)
                else:
                    beam = beam_generate(model, prompt['prefix_token_ids'], device, **options,
                                         num_beams=3 if setting['decoder'] == 'beam3' else 1,
                                         length_penalty=1.)
                    ids = beam['new']
                    record['sequence_log_probability'] = beam['score']
                record.update(status='completed', generated_token_ids=ids,
                              generated=tok.decode(ids, skip_special_tokens=True),
                              stopped_on_eos=bool(ids and ids[-1] == metadata['eos_id']))
            except Exception as exc:
                record.update(status='failed', error=repr(exc))
            result['records'].append(record)
            save()
    failed = sum(r['status'] == 'failed' for r in result['records'])
    result.update(status='completed' if not failed else 'completed_with_errors', failed_samples=failed)
    save()
    print(f"Saved {output / 'generations.txt'}; failed samples={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
