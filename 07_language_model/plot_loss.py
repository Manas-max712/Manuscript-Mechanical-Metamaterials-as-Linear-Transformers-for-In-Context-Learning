import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 with a.metrics.open() as f:rows=[r for r in csv.DictReader(f) if r['kind']=='epoch_end']
 if not rows:raise ValueError('No completed-epoch metrics found')
 epochs=[int(r['epoch']) for r in rows];train=[float(r['train_ce']) for r in rows];val=[float(r['val_ce']) for r in rows]
 with (a.output/'epoch_losses.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['epoch','training_cross_entropy','validation_cross_entropy']);w.writerows(zip(epochs,train,val))
 with plt.rc_context({'font.size':11,'axes.labelsize':13,'pdf.fonttype':42,'ps.fonttype':42}):
  fig,ax=plt.subplots(figsize=(6.2,4.2),layout='constrained')
  ax.plot(epochs,train,color='#2878B5',lw=2,marker='o',ms=3.5,label='Training')
  ax.plot(epochs,val,color='#D9534F',lw=2,marker='s',ms=3.5,label='Validation')
  ax.set_xlabel('Epoch');ax.set_ylabel('Cross entropy loss');ax.spines[['top','right']].set_visible(False)
  ax.grid(True,alpha=.18,lw=.6);ax.legend(frameon=False)
  for ext in ('png','pdf'):fig.savefig(a.output/f'Figure_S8.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig)
if __name__=='__main__':main()
