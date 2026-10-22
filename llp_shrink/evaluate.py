"""Evaluate a trusted training checkpoint on its original official test split."""
import argparse
import json
from pathlib import Path
import torch
from .config import Config
from .data import EvalDataset, eval_loader, raw_dataset, synthetic_dataset, fingerprint
from .runtime import accuracy
from .train import build_model, _device


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--root')
    p.add_argument('--device',default='auto')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--ema',action='store_true')
    args=p.parse_args(argv)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=Config(**ckpt['config'])
    cfg.device=args.device
    cfg.amp=False
    if args.root: cfg.root=args.root
    torch.set_num_threads(cfg.threads)
    device=_device(cfg)
    state=ckpt['ema_model'] if args.ema else ckpt['model']
    if state is None: raise ValueError('this checkpoint has no EMA model')
    model=build_model(cfg).to(device)
    model.load_state_dict(state,strict=True)
    if cfg.synthetic:
        images,labels=synthetic_dataset(cfg.dataset,cfg.synthetic_test,cfg.seed+1)
    else:
        images,labels=raw_dataset(cfg.dataset,cfg.root,False,False)
    if fingerprint(images,labels)!=ckpt['data_fingerprints']['test']:
        raise ValueError('test dataset fingerprint differs from training run')
    loader=eval_loader(EvalDataset(images,labels,cfg.dataset),cfg.eval_batch,args.workers,device.type=='cuda')
    result={'checkpoint':str(args.checkpoint),'epoch':ckpt['epoch'],'dataset':cfg.dataset,
            'ema':args.ema,'synthetic':cfg.synthetic,'test_samples':len(labels),
            'accuracy':accuracy(model,loader,device)}
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__': main()
