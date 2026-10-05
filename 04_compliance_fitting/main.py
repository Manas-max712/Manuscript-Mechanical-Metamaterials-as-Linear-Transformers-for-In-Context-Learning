#!/usr/bin/env python3
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from compliance_study.experiments import run
from compliance_study.tables import save_tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['all','s4','s5','s6'], default='all')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--smoke',action='store_true',help='Two starts, five iterations; workflow check only')
    args=parser.parse_args()
    output=(args.output_dir or Path('runs/smoke' if args.smoke else 'runs/paper')).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f'Output directory must be new or empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(target="inverse(Sigma)", study=args.study,smoke=args.smoke,seed=0,ny=4,objective='compliance',
                  theta_bound=6,maxiter=5 if args.smoke else 3000,ftol=1e-12,gtol=1e-9,
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    (output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    records=run(args.study,output,args.smoke)
    save_tables(records,args.study,output,args.smoke)
    (output/'COMPLETED.json').write_text(json.dumps({'groups':len(records)})+'\n')
    print(f'Saved results to {output}',flush=True)

if __name__=='__main__': main()
