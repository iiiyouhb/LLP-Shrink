import unittest
import numpy as np
import torch
from llp_shrink.data import BagPartition, BagDataset, EvalDataset, SPECS, synthetic_dataset, train_loader


class DataTests(unittest.TestCase):
    def test_augmentation_unsigned_integer_regressions(self):
        from llp_shrink.vendor import rand_rgb,rand_gray
        for module,shape in ((rand_rgb,(2,2,3)),(rand_gray,(2,2))):
            a=np.full(shape,20,dtype=np.uint8)
            a[1,1]=120
            with np.errstate(over='raise'):
                out=module.autocontrast_func(a)
            self.assertTrue(np.all(out[0,0]==0))
            self.assertTrue(np.all(out[1,1]==255))
            poster=module.posterize_func(np.full(shape,255,dtype=np.uint8),4)
            self.assertTrue(np.all(poster==240))

    def test_partition_disjoint_counts_repeatability(self):
        labels = np.arange(101)%10
        a = BagPartition.create(labels,16,3)
        b = BagPartition.create(labels,16,3)
        np.testing.assert_array_equal(a.indices,b.indices)
        self.assertEqual(len(np.unique(a.indices)),96)
        self.assertEqual(set(a.indices.flat)|set(a.omitted),set(range(101)))
        self.assertTrue((a.counts.sum(1)==16).all())
        for ids, counts in zip(a.indices,a.counts):
            np.testing.assert_array_equal(counts,np.bincount(labels[ids],minlength=10))
        self.assertFalse(np.array_equal(a.indices,BagPartition.create(labels,16,4).indices))

    def test_three_dataset_channels_transforms_no_label_leak(self):
        for name,(channels,size,_,_) in SPECS.items():
            images,labels = synthetic_dataset(name,32,3)
            part=BagPartition.create(labels,16,3)
            ds=BagDataset(images,part,name)
            for _ in range(4):
                b=ds[0]
                self.assertNotIn('audit_labels',b)
                for key in ('weak','strong'):
                    self.assertEqual(tuple(b[key].shape),(16,channels,size,size))
                    self.assertEqual(b[key].dtype,torch.float32)
                    self.assertTrue(torch.isfinite(b[key]).all())
            ev=EvalDataset(images,labels,name)
            self.assertTrue(torch.equal(ev[0][0],ev[0][0]))
            self.assertEqual(tuple(ev[0][0].shape),(channels,size,size))

    def test_worker_loader_and_batch_bag_order(self):
        images,labels=synthetic_dataset('FashionMNIST',36,3)
        p=BagPartition.create(labels,4,3)
        ds=BagDataset(images,p,'FashionMNIST',labels)
        for workers in (0,2):
            batches=list(train_loader(ds,2,workers,3,0))
            self.assertEqual(len(batches),4)
            for b in batches:
                for ids,truth,counts in zip(b['ids'],b['audit_labels'],b['counts']):
                    np.testing.assert_array_equal(truth,labels[ids])
                    np.testing.assert_array_equal(counts,np.bincount(truth,minlength=10))


if __name__ == '__main__': unittest.main()
