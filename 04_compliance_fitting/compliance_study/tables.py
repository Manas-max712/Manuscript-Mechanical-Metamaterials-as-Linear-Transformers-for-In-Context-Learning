from .experiments import FAMILIES, TOPOLOGIES, write_csv
NAMES = dict(identity='Isotropic', diag='Diagonal', rotated='Rotated')
LABELS = dict(zip(TOPOLOGIES, ['Locally connected','Shallow','Medium','Large','Very large','Complete graph']))


def export(output, number, rows, caption):
    path = output / 'tables'; path.mkdir(exist_ok=True)
    write_csv(path/f'Table_{number}.csv',rows)
    keys = list(rows[0])
    def fmt(value):
        if isinstance(value,float):
            m,e=f'{value:.2e}'.split('e')
            return rf'${m}\times10^{{{int(e)}}}$'
        return str(value)
    text = '\\begin{table}[t]\n\\centering\n\\small\n\\caption{'+caption+'}\n'
    text += '\\begin{tabular}{'+'l'*len(keys)+'}\n\\hline\n'
    text += ' & '.join(keys)+r' \\'+'\n\\hline\n'
    text += ''.join(' & '.join(fmt(v) for v in row.values())+r' \\'+'\n' for row in rows)
    text += '\\hline\n\\end{tabular}\n\\end{table}\n'
    (path/f'Table_{number}.tex').write_text(text)


def save_tables(records, study, output, smoke=False):
    def get(d,f,p,n): return records[(d,f,p,2 if smoke else n)]
    metric = r'Errors are relative compliance errors $\|G_{\mathrm{phys}}-G_{\mathrm{ideal}}\|_F/\|G_{\mathrm{ideal}}\|_F$. '
    note = 'Workflow check only: two starts and five iterations per start. ' if smoke else ''
    if study in ('all','s4'):
        rows=[{'Task family':NAMES[f], **{f'd = {d}':get(d,f,'baseline',50)['best'] for d in (5,10,15,20)}} for f in FAMILIES]
        export(output,'S4',rows,note+'Dimension scaling at fixed lattice height $n_y=4$. '+metric)
    if study in ('all','s5'):
        rows=[]
        for p in TOPOLOGIES:
            for n in ([50,150] if p=='complete' and not smoke else [50]):
                r=get(10,'rotated',p,n)
                rows.append({'Topology':LABELS[p],'Bars':r['bars'],'Initializations':r['initializations'],'Compliance fit error':r['best']})
        export(output,'S5',rows,note+'Connectivity study at $d=10$ with 50 nodes. '+metric)
    if study in ('all','s6'):
        rows=[]
        for d in (5,10):
            for f in FAMILIES:
                for n in ([50,150] if f=='rotated' and not smoke else [50]):
                    rows.append(get(d,f,'baseline',n))
        for n in ([50,100,150,200] if not smoke else [50]): rows.append(get(10,'rotated','complete',n))
        rows=[{'$d$':r['d'],'Task family':NAMES[r['family']],'Bars':r['bars'],'Initializations':r['initializations'],'Best':r['best'],'Median':r['median'],'Worst':r['worst']} for r in rows]
        export(output,'S6',rows,note+'Multi-start diagnostics for direct compliance fitting. '+metric)
