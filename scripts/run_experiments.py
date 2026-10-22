"""Print or execute the Lu.pdf main/threshold grid, one process per experiment."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]


def build_jobs(suite,seeds,thresholds):
    datasets=['CIFAR10','SVHN','FashionMNIST'] if suite=='main' else ['CIFAR10']
    return [dict(dataset=ds,bag_size=m,seed=seed,threshold=t)
            for ds in datasets for m in (16,128) for seed in seeds
            for t in ([.95] if suite=='main' else thresholds)]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['main','threshold'],default='main')
    p.add_argument('--seeds',type=int,nargs='+',default=[3],help='AHIL default is seed=3; multiple runs must be explicit')
    p.add_argument('--thresholds',type=float,nargs='+',default=[0,.25,.5,.75,.95,.96,.98,1],
                   help='candidate grid read from figure, not a recovered author sweep')
    p.add_argument('--root',default='data')
    p.add_argument('--output-root',default='runs')
    p.add_argument('--device',default='cuda')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--execute',action='store_true',help='without this flag only print commands')
    p.add_argument('--resume',action='store_true',help='resume incomplete existing runs, skip complete ones')
    p.add_argument('--download',action='store_true')
    p.add_argument('--amp',action='store_true')
    args=p.parse_args(argv)
    args.root=str(Path(args.root).resolve())
    args.output_root=str(Path(args.output_root).resolve())
    if any(not 0<=t<=1 for t in args.thresholds): p.error('thresholds must be in [0,1]')
    jobs=build_jobs(args.suite,args.seeds,args.thresholds)
    for job in jobs:
        out=Path(args.output_root)/job['dataset']/f"shrink_M{job['bag_size']}_s{job['seed']}_tau{job['threshold']:g}"
        cmd=[sys.executable,'-m','llp_shrink.train','--config',str(PROJECT/'configs/paper.json'),
             '--root',args.root,'--output',str(out),'--device',args.device,'--workers',str(args.workers)]
        for key,value in job.items(): cmd += ['--'+key.replace('_','-'),str(value)]
        if args.download: cmd.append('--download')
        if args.amp: cmd.append('--amp')
        if args.resume and (out/'last.pth').exists():
            summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
            if summary.get('complete'):
                print('SKIP complete:',out)
                continue
            cmd += ['--resume',str(out/'last.pth')]
        print(shlex.join(cmd),flush=True)
        if args.execute:
            subprocess.run(cmd,check=True,cwd=PROJECT)


if __name__=='__main__': main()
