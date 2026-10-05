import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=10)
    p.add_argument('--figure-steps', type=int, default=2)
    p.add_argument('--require-cuda', action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(a.package_dir.resolve()/'src'))
    import numpy as np
    import torch
    from transformers import GPT2TokenizerFast
    from common import make_model, sha256, atomic_json, object_hash, SCHEMA
    from gd_build import build_geometry
    import paper_plot_helpers as h
    torch.set_num_threads(8)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.require_cuda and device.type!='cuda': raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'checkpoint_used.pt'
    shutil.copyfile(a.checkpoint,snapshot)
    c=torch.load(snapshot,map_location='cpu',weights_only=False)
    manifest=json.loads((a.data/'manifest.json').read_text())
    if 'files' in manifest and object_hash({'settings':manifest['settings'], 'files':manifest['files']})!=manifest['dataset_id']:
        raise RuntimeError('Manifest identity mismatch')
    if c.get('schema')!=SCHEMA or c['dataset_id']!=manifest['dataset_id']:
        raise RuntimeError('Checkpoint/data mismatch')
    model=make_model(c['model_config'],device)
    model.load_state_dict(c['model'],strict=True)
    model.eval()
    tok=GPT2TokenizerFast.from_pretrained(a.data/'tokenizer',local_files_only=True)
    assert len(tok)==manifest['vocab_size']==c['model_config']['vocab_limit']
    assert tok.pad_token_id==manifest['pad_id'] and tok.eos_token_id==manifest['eos_id']
    geom=build_geometry(SimpleNamespace(**c['model_config']))
    nodes=np.asarray(geom.nodes); edges=np.asarray(geom.edges)
    ports=[int(d)//2 for d in geom.out_dofs[:model.n_ctx+1]]
    prompts=[
      'Once upon a time, there was a little boy named Tim. Tim had a big box of toys that he needed to unpack',
      'Once upon a time, there was a quiet little dog named Spot. Spot loved to play with his',
      'In a tiny world full of wonders, lived minuscule creatures called microbes. They were too small to be seen by our eyes, but they had',
      'Once upon a time, in a small village named Harmonyville, lived two best friends, Sam and Alex. They loved exploring old buildings',
    ]
    result=dict(checkpoint=str(a.checkpoint),sha256=sha256(snapshot),epoch=c['epoch'],
      global_step=c['global_step'],next_batch=c.get('next_batch'),model_config=c['model_config'],
      decoding='greedy; no repetition penalty',records=[],head=0,layer=len(model.W_o)-1,
      scope='Configure: fixed Newton approximation. Read: one raw tangent value channel, before causal RMS normalization and electronic mixing.')
    atomic_json(a.output/'generations.json',result)
    fields=[]
    with torch.no_grad():
      for pi,prompt in enumerate(prompts):
        cur=tok.encode(prompt,add_special_tokens=False); new=[]
        for step in range(a.tokens):
          ids=torch.tensor([cur[-model.n_ctx:]],device=device)
          logits=model(ids)
          if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite logits')
          scores=logits[0,-1].clone()
          scores[manifest['pad_id']]=-torch.inf
          nxt=int(scores.argmax())
          if pi==0 and step<a.figure_steps:
            f=h._geo_frontier_config_fields(model,ids,len(model.W_o)-1)
            S=ids.shape[1]
            rd,reaction,channel,colors,free=h._geo_settle_and_read(model,geom,nodes,f['V'],f['u_heads'][0],f['k'][0],f['prestretch'][0],S,-1)
            cfg=np.zeros(2*len(nodes)); cfg[np.asarray(geom.free)]=free
            fields.append(dict(S=S,rd=rd,reaction=reaction,channel=channel,colors=colors,cfg=cfg.reshape(-1,2),token=tok.decode([nxt])))
          new.append(nxt);cur.append(nxt)
          if nxt==manifest['eos_id']: break
        result['records'].append(dict(prompt=prompt,generated=tok.decode(new,skip_special_tokens=True),token_ids=new))
        atomic_json(a.output/'generations.json',result)
        print(prompt+' '+result['records'][-1]['generated'],flush=True)
    (a.output/'generations.txt').write_text('\n\n'.join('PROMPT: '+r['prompt']+'\nCONTINUATION: '+r['generated'] for r in result['records']))
    keep=np.ones(len(edges),dtype=bool)
    limit=max(float(np.percentile(np.abs(np.concatenate([f['colors'] for f in fields])),96)),1e-9)
    norm=h.matplotlib.colors.TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    wn=h.matplotlib.colors.Normalize(0,limit); cmap=h.matplotlib.colormaps['coolwarm']
    cs=.72/max(max(float(np.abs(f['cfg']).max()) for f in fields),1e-9)
    ds=.30/max(max(float(np.abs(f['rd']).max()) for f in fields),1e-9)
    rs=1.2/max(max(abs(float(f['reaction'][f['S']])) for f in fields),1e-9)
    for i,f in enumerate(fields):
      if not all(np.isfinite(f[k]).all() for k in ('cfg','rd','reaction','colors')):
        raise FloatingPointError('Nonfinite mechanical fields')
      np.savez(a.output/f'step{i}_fields.npz',**f)
      for zoom,suffix in ((True,''),(False,'_full')):
        for ext in ('png','pdf'):
          h._plot_geo_configure(nodes,edges,keep,ports,f['S'],f['cfg'],f['colors'],norm,cmap,
            a.output/f'step{i}_configure{suffix}.{ext}',cs,zoom=zoom,wvals=f['colors'],wnorm=wn)
          h._plot_read(nodes,edges,keep,ports,f['S'],f['rd'],f['reaction'],f['colors'],norm,cmap,
            a.output/f'step{i}_read{suffix}.{ext}',ds,rs,zoom=zoom,wvals=f['colors'],wnorm=wn,cfg_disp=f['cfg'],cfg_scale=cs)
    h._save_colorbar(cmap,norm,a.output/'colorbar.pdf',r'Signed geometric stiffness $t_e/L_e$')
    fig,axes=h.plt.subplots(2,len(fields),figsize=(6*len(fields),7),squeeze=False)
    for i,f in enumerate(fields):
      for row,phase in enumerate(('configure','read')):
        axes[row,i].imshow(h.plt.imread(a.output/f'step{i}_{phase}.png'))
        axes[row,i].axis('off')
        axes[row,i].set_title(f'{phase.capitalize()} {i+1}' + (f" → {f['token']!r}" if row else ''))
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(a.output/f'Figure_6_storyboard.{ext}',dpi=200,bbox_inches='tight')
    h.plt.close(fig)
    result.update(status='completed',configure_display_scale=cs,read_display_scale=ds)
    atomic_json(a.output/'generations.json',result)
    print(f'Completed: {a.output}',flush=True)

if __name__=='__main__': main()
