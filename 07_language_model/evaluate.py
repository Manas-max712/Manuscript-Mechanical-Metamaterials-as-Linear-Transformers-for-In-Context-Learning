import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import torch
from common import SCHEMA,make_model,atomic_json,sha256
from gd_data import TokenDataset,load_manifest
from gd_eval import evaluate
from causality import cpu_checkpoint_causality

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
 torch.set_num_threads(8)
 c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=load_manifest(a.data)
 if c.get('schema')!=SCHEMA or c['dataset_id']!=m['dataset_id']:raise ValueError('Checkpoint/data mismatch')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False
 model=make_model(c['model_config'],device);model.load_state_dict(c['model']);model.eval()
 result=dict(checkpoint_sha256=sha256(a.checkpoint),metrics={s:evaluate(model,TokenDataset(a.data,s),a.batch_size,device) for s in ('valid','test')})
 atomic_json(a.output,result)
 ids,length=TokenDataset(a.data,'valid')[0]
 result['causality']=cpu_checkpoint_causality(c['model_config'],c['model'],ids[:length-1][None])
 atomic_json(a.output,result)
if __name__=='__main__':main()
