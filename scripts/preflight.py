"""One real-data WRN-28-2 update and reload check; NOT a full experiment."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from llp_shrink.config import Config
from llp_shrink.data import raw_dataset, BagPartition, BagDataset, EvalDataset, eval_loader
from llp_shrink.losses import shrink_loss
from llp_shrink.train import build_model
from llp_shrink.runtime import seed_all, accuracy, environment


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='FashionMNIST',choices=['CIFAR10','SVHN','FashionMNIST'])
    p.add_argument('--root',default='data')
    p.add_argument('--download',action='store_true')
    p.add_argument('--bag-size',type=int,default=16)
    p.add_argument('--bags',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    seed_all(3)
    torch.set_num_threads(4)
    images,labels=raw_dataset(args.dataset,args.root,True,args.download)
    test_images,test_labels=raw_dataset(args.dataset,args.root,False,args.download)
    partition=BagPartition.create(labels,args.bag_size,3)
    ds=BagDataset(images,partition,args.dataset)
    batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=args.bags)))
    cfg=Config(dataset=args.dataset)
    model=build_model(cfg).to(args.device).train()
    initial=model.classifier.weight.detach().clone()
    optim=torch.optim.SGD(model.parameters(),lr=.03,momentum=.9,nesterov=True,weight_decay=.0005)
    weak,strong=[batch[k].to(args.device).flatten(0,1) for k in ('weak','strong')]
    weak_logits,strong_logits=model(torch.cat([weak,strong])).chunk(2)
    shape=(args.bags,args.bag_size,10)
    # Test paper tau AND an intentionally low tau to exercise the selected branch.
    outputs=[]
    for tau in (.95,0.):
        result=shrink_loss(weak_logits.reshape(shape),strong_logits.reshape(shape),batch['counts'].to(args.device),tau)
        outputs.append(dict(threshold=tau,loss=result.total.item(),selected=result.selected.sum().item()))
    result.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
    optim.step()
    if torch.equal(initial,model.classifier.weight): raise AssertionError('parameters did not change')
    test_ds=EvalDataset(test_images[:19],test_labels[:19],args.dataset)
    loader=eval_loader(test_ds,8,0)
    acc=accuracy(model,loader,torch.device(args.device))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'weights.pt'
        torch.save(model.state_dict(),path)
        restored=build_model(cfg).to(args.device)
        restored.load_state_dict(torch.load(path,map_location=args.device,weights_only=True))
        acc2=accuracy(restored,loader,torch.device(args.device))
        if acc!=acc2: raise AssertionError('reload accuracy changed')
    report=dict(kind='real-data single-step preflight; NOT paper accuracy',dataset=args.dataset,
                original_train_samples=len(images),original_test_samples=len(test_labels),
                train_samples_used=args.bags*args.bag_size,test_samples_used=len(test_ds),
                architecture='WRN-28-2',checks=outputs,parameter_update=True,reload_equal=True,
                environment=environment())
    text=json.dumps(report,indent=2,allow_nan=False)
    print(text)
    if args.output: args.output.write_text(text+'\n',encoding='utf-8')


if __name__=='__main__': main()
