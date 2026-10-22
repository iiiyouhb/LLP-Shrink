"""Summarize actual completed runs; never fill absent cells with paper results."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root,include_synthetic=False):
    groups=defaultdict(list)
    for path in sorted(Path(root).rglob('summary.json')):
        summary=json.loads(path.read_text(encoding='utf-8'))
        if not summary.get('complete') or (summary.get('synthetic') and not include_synthetic): continue
        cfg=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        # Every numerical/augmentation/training setting except seed and output must match.
        exclude={'seed','output','root','device','workers','download','log_every','save_every','export_weights_every','threads'}
        key=json.dumps({k:v for k,v in cfg.items() if k not in exclude},sort_keys=True)
        groups[key].append((cfg,summary,path.parent))
    rows=[]
    for key,runs in groups.items():
        cfg=runs[0][0]
        seeds=[r[0]['seed'] for r in runs]
        if len(seeds)!=len(set(seeds)):
            raise ValueError(f'duplicate seed runs in group: {seeds}; choose a clean output root')
        row={'dataset':cfg['dataset'],'method':cfg['method'],'bag_size':cfg['bag_size'],
             'threshold':cfg['threshold'],'seeds':seeds,'n':len(runs),'synthetic':cfg['synthetic'],
             'settings':json.loads(key),'runs':[str(r[2]) for r in runs]}
        for metric in ('raw_acc','ema_acc'):
            values=[r[1]['final'][metric] for r in runs if r[1]['final'][metric] is not None]
            row[metric]={'mean':statistics.mean(values) if values else None,
                         'sample_std':statistics.stdev(values) if len(values)>1 else None}
        rows.append(row)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--include-synthetic',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    text=json.dumps(summarize(args.root,args.include_synthetic),indent=2,ensure_ascii=False,allow_nan=False)
    if args.output: args.output.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__': main()
