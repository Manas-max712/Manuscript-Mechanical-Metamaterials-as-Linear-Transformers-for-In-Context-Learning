from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .config import LAYERS


def export_extraction_diagram(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(-.8, 10.4)
    ax.set_ylim(-1.9, 1.5)
    ax.axis('off')
    boxes = [(0, r'$F=e_i$', '#eeeeee')]
    boxes += [(2+2*i, f'Layer {i+1}', '#dbeafe') for i in range(LAYERS)]
    boxes += [(8, r'$w^{(i)}$', '#eeeeee')]
    for x, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x,-.35), 1.2, .85, boxstyle='round,pad=0.08',
                                   facecolor=color, edgecolor='black'))
        ax.text(x+.6,.075,label,ha='center',va='center',fontsize=16)
    for (x,_,_), (next_x,_,_) in zip(boxes, boxes[1:]):
        ax.annotate('',xy=(next_x-.1,.075),xytext=(x+1.3,.075),
                    arrowprops={'arrowstyle':'->','linewidth':1.5})
    ax.text(4.6,1.1,'Trained lattice Transformer',ha='center',fontsize=17)
    ax.text(4.6,-1.25,r'$G_{\mathrm{LT}}=[\,w^{(1)}\;w^{(2)}\;\cdots\;w^{(d)}\,],'
            r'\qquad w^{(i)}=G_{\mathrm{LT}}e_i$',ha='center',fontsize=19)
    for ext in ('png','pdf'):
        fig.savefig(output_dir/f'Figure_2a.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
