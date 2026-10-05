import argparse,csv,json,math
from pathlib import Path

def tex(text):
 mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
 return ''.join(mapping.get(c,c) for c in str(text))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--study',type=Path,required=True);p.add_argument('--generations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=[]
 for n in (1,2):
  run=a.study/f'newton{n}'
  if not json.loads((run/"summary.json").read_text()).get("training_complete", False):
   raise ValueError(f"Training is incomplete: {run}")
  with (run/'metrics.csv').open() as f:history=list(csv.DictReader(f))
  ce=min(float(r['val_ce']) for r in history if r['val_ce'])
  residual=json.loads((a.study/'diagnostics'/f'best_newton{n}.json').read_text())
  best=min((r for r in history if r['val_ce']), key=lambda r:float(r['val_ce']))
  if residual['newton_steps'] != n or residual['global_step'] != int(best['global_step']):
   raise ValueError('Residual measurement must use the best-validation checkpoint for this arm')
  rows.append(dict(newton_steps=n,best_validation_ce=ce,validation_perplexity=math.exp(ce),mean_initial_force_norm=residual['initial_force_norm']['mean'],mean_final_force_norm=residual['final_force_norm']['mean'],median_relative_residual=residual['final_over_initial']['median']))
 with (a.output/'Table_S13.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 labels=['Best validation cross-entropy','Validation perplexity','Mean initial force norm','Mean final force norm','Median relative residual']
 lines=[r'\begin{tabular}{lrr}',r'\hline',r'Metric & One Newton update & Two Newton updates \\',r'\hline']
 for key,label in zip(list(rows[0])[1:],labels):lines.append(f'{label} & {rows[0][key]:.6g} & {rows[1][key]:.6g}'+r' \\')
 lines.extend([r'\hline',r'\end{tabular}']);(a.output/'Table_S13.tex').write_text('\n'.join(lines)+'\n')
 records=json.loads(a.generations.read_text())['records']
 selected=[r for r in records if any(name in r['prompt'] for name in ['Spot','microbes','Harmonyville'])]
 if len(selected)!=3:raise ValueError('Expected the three paper prompts')
 (a.output/'Table_S14.json').write_text(json.dumps(selected,indent=2)+'\n')
 lines=[r'\begin{tabular}{p{0.44\linewidth}p{0.48\linewidth}}',r'\hline',r'Prompt & Continuation \\',r'\hline']
 for r in selected:lines.append(tex(r['prompt'])+' & '+tex(r['generated'])+r' \\')
 lines.extend([r'\hline',r'\end{tabular}']);(a.output/'Table_S14.tex').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
