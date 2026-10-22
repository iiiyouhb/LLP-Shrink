import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from llp_shrink.config import Config
from llp_shrink.runtime import accuracy, lr_factor, EMA
from llp_shrink.train import run, build_model


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_test_denominator_incomplete_last_batch(self):
        class Identity(torch.nn.Module):
            def forward(self,x): return x
        logits=torch.tensor([[1.,0.]]*5)
        labels=torch.tensor([0,0,0,0,1])
        loader=DataLoader(TensorDataset(logits,labels),batch_size=4)
        self.assertEqual(accuracy(Identity(),loader,torch.device('cpu')),80.)

    def test_wrn28_channels_and_parameter_update(self):
        for dataset,channels,size in [('CIFAR10',3,32),('SVHN',3,32),('FashionMNIST',1,28)]:
            cfg=Config(dataset=dataset)
            model=build_model(cfg)
            before=model.classifier.weight.detach().clone()
            optim=torch.optim.SGD(model.parameters(),lr=.01)
            out=model(torch.randn(2,channels,size,size))
            self.assertEqual(tuple(out.shape),(2,10))
            torch.nn.functional.cross_entropy(out,torch.tensor([0,1])).backward()
            optim.step()
            self.assertFalse(torch.equal(before,model.classifier.weight))

    def test_scheduler_matches_upstream_active_definition(self):
        self.assertEqual(lr_factor(0,100,'upstream'),1)
        self.assertAlmostEqual(lr_factor(50,100,'upstream'),2**-.5)
        self.assertAlmostEqual(lr_factor(100,100,'upstream'),0)

    def test_ema_copies_buffers(self):
        m=torch.nn.BatchNorm1d(2)
        ema=EMA(m,.5)
        with torch.no_grad():
            m.weight.fill_(3)
            m.running_mean.fill_(7)
            m.num_batches_tracked.fill_(10)
        ema.update(m)
        torch.testing.assert_close(ema.model.weight,torch.full((2,),2.))
        torch.testing.assert_close(ema.model.running_mean,torch.full((2,),7.))
        self.assertEqual(ema.model.num_batches_tracked.item(),10)

    def test_epoch_resume_exact_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            common=dict(dataset='FashionMNIST',synthetic=True,synthetic_train=16,synthetic_test=11,
                        bag_size=4,batch_instances=8,epochs=2,depth=10,width=1,workers=0,
                        device='cpu',threads=2,threshold=0,export_weights_every=1,save_every=1,
                        eval_batch=6,audit_labels=True)
            full=Path(tmp)/'full'; resumed=Path(tmp)/'resumed'
            with contextlib.redirect_stdout(io.StringIO()):
                run(Config(output=str(full),**common))
                cfg=Config(output=str(resumed),**common)
                run(cfg,stop_after_epoch=1)
                self.assertFalse(json.loads((resumed/'summary.json').read_text())['complete'])
                run(cfg,resume=resumed/'last.pth')
            a=torch.load(full/'last.pth',weights_only=False)
            b=torch.load(resumed/'last.pth',weights_only=False)
            self.assertEqual(a['global_step'],4)
            self.assertEqual(a['global_step'],b['global_step'])
            for key in ('model','ema_model'):
                for name,value in a[key].items():
                    self.assertTrue(torch.equal(value,b[key][name]),f'{key}.{name}')
            for ra,rb in zip(a['history'],b['history']):
                self.assertEqual({k:v for k,v in ra.items() if k!='seconds'},
                                 {k:v for k,v in rb.items() if k!='seconds'})
            import numpy as np
            with np.load(resumed/'sample_weights_epoch_0001.npz') as weights:
                self.assertEqual(weights['ids'].size,16)
                self.assertEqual(len(np.unique(weights['ids'])),16)
                self.assertTrue(np.isfinite(weights['confidence']).all())
                self.assertTrue(np.array_equal(weights['weights'],weights['selected'].astype(float)))
            self.assertTrue((resumed/'model_final.pt').exists())
            self.assertTrue((resumed/'best_raw.pth').exists())
            self.assertTrue((resumed/'best_ema.pth').exists())
            self.assertEqual(len((resumed/'metrics.jsonl').read_text().splitlines()),2)
            self.assertFalse((resumed/'.running.lock').exists())
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    run(Config(output=str(resumed),**dict(common,threshold=.5)),resume=resumed/'last.pth')

    def test_no_ema_and_reference_methods_smoke(self):
        for method in ('shrink','dllp','ahil','supervised'):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg=Config(output=tmp,dataset='CIFAR10',synthetic=True,synthetic_train=8,synthetic_test=5,
                           bag_size=4,batch_instances=8,epochs=1,depth=10,width=1,workers=0,
                           device='cpu',threads=2,ema=False,method=method)
                rows=run(cfg)
                self.assertIsNone(rows[0]['ema_acc'])
                self.assertTrue((Path(tmp)/'last.pth').exists())

    def test_diagnostics_do_not_change_training_and_existing_run_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            paths=[Path(tmp)/'without_labels',Path(tmp)/'with_audit_labels']
            for path,audit in zip(paths,(False,True)):
                cfg=Config(output=str(path),synthetic=True,synthetic_train=8,synthetic_test=5,
                           batch_instances=8,bag_size=4,epochs=1,depth=10,width=1,
                           workers=0,threads=2,audit_labels=audit,device='cpu',threshold=0)
                run(cfg)
                before=(path/'config.json').read_bytes()
                with self.assertRaises(FileExistsError): run(cfg)
                self.assertEqual(before,(path/'config.json').read_bytes())
            a=torch.load(paths[0]/'model_final.pt',weights_only=True)
            b=torch.load(paths[1]/'model_final.pt',weights_only=True)
            for key in a: self.assertTrue(torch.equal(a[key],b[key]),key)


if __name__=='__main__': unittest.main()
