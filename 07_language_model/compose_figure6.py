from pathlib import Path
import sys, json, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/figure6mpl')
import numpy as np
from types import SimpleNamespace
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
from matplotlib.patches import FancyArrowPatch
import argparse
parser=argparse.ArgumentParser(description="Compose Figure 6 without token labels on mechanical panels.")
parser.add_argument('--results',type=Path,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
OUT=args.results.resolve()
sys.path.insert(0,str(ROOT/'src'))
from gd_build import build_geometry
import paper_plot_helpers as h
r=json.loads((OUT/'generations.json').read_text()); cfg=r['model_config']
g=build_geometry(SimpleNamespace(**cfg)); nodes=np.asarray(g.nodes); edges=np.asarray(g.edges)
ports=[int(x)//2 for x in g.out_dofs[:65]]; keep=np.ones(len(edges),bool)
f=[dict(np.load(OUT/f'step{i}_fields.npz')) for i in range(2)]
lim=max(float(np.percentile(np.abs(np.concatenate([x['colors'] for x in f])),96)),1e-9)
norm=h.matplotlib.colors.TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim); wn=h.matplotlib.colors.Normalize(0,lim); cmap=h.matplotlib.colormaps['coolwarm']
cs=.72/max(float(abs(x['cfg']).max()) for x in f); ds=.30/max(float(abs(x['rd']).max()) for x in f); rs=1.2/max(abs(float(x['reaction'][int(x['S'])])) for x in f)
h._legend=lambda *args,**kwargs: None
for i,x in enumerate(f):
 S=int(x['S'])
 for ext in ('png','pdf'):
  h._plot_geo_configure(nodes,edges,keep,ports,S,x['cfg'],x['colors'],norm,cmap,OUT/f'panel_{i}_configure.{ext}',cs,zoom=True,wvals=x['colors'],wnorm=wn,token_texts=None)
  h._plot_read(nodes,edges,keep,ports,S,x['rd'],x['reaction'],x['colors'],norm,cmap,OUT/f'panel_{i}_read.{ext}',ds,rs,zoom=True,wvals=x['colors'],wnorm=wn,token_texts=None,next_label=None,cfg_disp=x['cfg'],cfg_scale=cs)
x=f[0]
h._plot_geo_configure(nodes,edges,keep,ports,int(x['S']),x['cfg'],x['colors'],norm,cmap,OUT/'panel_full.png',cs,zoom=False,wvals=x['colors'],wnorm=wn)
fig=h.plt.figure(figsize=(14,8),facecolor='white')
positions=[(.01,.22,.35,.58),(.37,.48,.29,.35),(.69,.48,.29,.35),(.69,.08,.29,.35),(.37,.08,.29,.35)]
files=['panel_full.png','panel_0_configure.png','panel_0_read.png','panel_1_configure.png','panel_1_read.png']
labels=['','Stage I','Stage II','Stage I','Stage II']
for pos,file,label in zip(positions,files,labels):
 ax=fig.add_axes(pos); ax.imshow(h.plt.imread(OUT/file)); ax.axis('off')
 if label: ax.text(.5,-.01,label,transform=ax.transAxes,ha='center',fontsize=12)
fig.text(.37,.95,'Once upon a time, there was a little boy named Tim. Tim had a big',fontsize=13)
box=HPacker(children=[TextArea('box of toys that he needed to unpack',textprops={'fontsize':13}),TextArea(r['records'][0]['generated'],textprops={'fontsize':13,'color':'red'})],align='baseline',pad=0,sep=0)
anch=AnchoredOffsetbox(loc='upper left',child=box,frameon=False,bbox_to_anchor=(.37,.935),bbox_transform=fig.transFigure,borderpad=0,pad=0);fig.add_artist(anch)
for a,b in [((.33,.61),(.38,.66)),((.655,.65),(.69,.65)),((.835,.47),(.835,.43)),((.69,.25),(.655,.25))]:
 fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='simple',mutation_scale=22,color='black'))
fig.text(.02,.035,'Head 1, layer 2; displacements magnified. Red text: greedy model output.',fontsize=10)
for ext in ('png','pdf'):fig.savefig(OUT/f'Figure_6.{ext}',dpi=300,bbox_inches='tight')
h.plt.close(fig)
(OUT/'caption_unlabelled.tex').write_text(r'''\caption{Prompt-dependent geometric-prestress computation for story generation using the trained one-Newton-step model. Black text is the supplied prompt; red text is the ten-token greedy continuation. The panels illustrate the first two generated tokens. Stage I applies prompt-dependent port displacements and one regularized Newton update to the interior coordinates. Stage II illustrates a tangent response at this approximate configuration. The model combines the mechanical heads with causal normalization and digital readout to predict the next token, which is appended before the cycle repeats. The mechanical panels show one head in the second layer and one value channel; displacements are magnified for visibility.}''')
