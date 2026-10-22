import math
import unittest
import torch
import torch.nn.functional as F
from llp_shrink.losses import shrink_loss, reference_loss, quota_selection


class LossTests(unittest.TestCase):
    def test_hand_calculated_class_specific_sets(self):
        p = torch.tensor([[[.98, .01, .01], [.02, .97, .01], [.80, .10, .10], [.1, .2, .7]]], dtype=torch.float64)
        strong = torch.tensor([[[2., 0., 0.], [0., 3., 0.], [1., 1., 0.], [1., 0., 1.]]], dtype=torch.float64, requires_grad=True)
        weak = p.log().requires_grad_()
        r = shrink_loss(weak, strong, torch.tensor([[2, 1, 1]]), .95)
        self.assertEqual(r.selected.tolist(), [[True, True, False, False]])
        self.assertEqual(r.remaining.tolist(), [[3, 3, 4]])
        torch.testing.assert_close(r.targets, torch.tensor([[1/3, 0, 1/4]], dtype=torch.float64))
        expected_pred = torch.tensor([[(.02+.8+.1)/3, (.01+.1+.2)/3, .82/4]], dtype=torch.float64)
        torch.testing.assert_close(r.predictions, expected_pred)
        expected_b = -math.log((.02+.8+.1)/3)/3 - math.log(.82/4)/4
        expected_i = (math.log(1+2*math.exp(-2)) + math.log(1+2*math.exp(-3)))/2
        self.assertAlmostEqual(r.bag.item(), expected_b, places=12)
        self.assertAlmostEqual(r.instance.item(), expected_i, places=12)
        self.assertAlmostEqual(r.total.item(), .5*(expected_b+expected_i), places=12)
        self.assertFalse(r.targets.requires_grad)
        self.assertFalse(r.ratio.requires_grad)
        r.total.backward()
        self.assertTrue(torch.isfinite(weak.grad).all())
        self.assertTrue(torch.isfinite(strong.grad).all())
        torch.testing.assert_close(strong.grad[0, 2:], torch.zeros_like(strong.grad[0, 2:]))

    def test_quota_highest_confidence_and_stable_ties(self):
        p = torch.tensor([[[.96,.04],[.99,.01],[.98,.02],[.99,.01]]])
        selected, _, _ = quota_selection(p, torch.tensor([[2,2]]), .95)
        self.assertEqual(selected.tolist(), [[False,True,False,True]])
        selected, _, _ = quota_selection(p, torch.tensor([[1,3]]), .95)
        self.assertEqual(selected.tolist(), [[False,True,False,False]])

    def test_strict_threshold_and_zero_quota(self):
        p = torch.tensor([[[.75,.25],[.99,.01]]], dtype=torch.float64)
        selected, _, _ = quota_selection(p, torch.tensor([[2,0]]), .75)
        self.assertEqual(selected.tolist(), [[False,True]])
        selected, _, _ = quota_selection(p, torch.tensor([[0,2]]), 0)
        self.assertFalse(selected.any())

    def test_none_selected_equals_dllp(self):
        weak = torch.randn(2,4,3, requires_grad=True)
        strong = torch.randn_like(weak, requires_grad=True)
        counts = torch.tensor([[2,1,1],[0,2,2]])
        r = shrink_loss(weak,strong,counts,1)
        expected = -(counts/4 * weak.softmax(-1).mean(1).log()).sum(-1).mean()
        torch.testing.assert_close(r.total, expected)
        self.assertEqual(r.instance.item(), 0)
        self.assertEqual(r.ratio.item(), 0)
        r.total.backward()
        torch.testing.assert_close(strong.grad, torch.zeros_like(strong))

    def test_all_selected_and_empty_residual_set(self):
        weak = torch.tensor([[[1000.,-1000.],[1000.,-1000.]]], requires_grad=True)
        strong = torch.zeros_like(weak, requires_grad=True)
        r = shrink_loss(weak,strong,torch.tensor([[2,0]]),.95)
        self.assertEqual(r.remaining.tolist(), [[0,2]])
        self.assertEqual(r.bag.item(),0)
        self.assertEqual(r.ratio.item(),1)
        self.assertAlmostEqual(r.total.item(),math.log(2),places=6)
        r.total.backward()
        self.assertTrue(torch.isfinite(weak.grad).all())
        self.assertTrue(torch.isfinite(strong.grad).all())

    def test_per_bag_mean_includes_empty_bags(self):
        weak = torch.tensor([[[6.,0.],[0.,6.]],[[0.,0.],[0.,0.]]])
        strong = torch.zeros_like(weak)
        r = shrink_loss(weak,strong,torch.tensor([[1,1],[1,1]]),.95)
        self.assertAlmostEqual(r.instance.item(),math.log(2)/2,places=6)
        self.assertEqual(r.ratio.item(),.5)

    def test_random_invariants_and_extreme_gradients(self):
        for seed in range(12):
            torch.manual_seed(seed)
            weak = (torch.randn(3,16,4)* (1000 if seed==0 else 4)).requires_grad_()
            strong = torch.randn_like(weak,requires_grad=True)
            counts = F.one_hot(torch.randint(4,(3,16)),4).sum(1)
            r = shrink_loss(weak,strong,counts,seed/12)
            removed = (F.one_hot(r.pseudo,4)*r.selected.unsqueeze(-1)).sum(1)
            self.assertTrue((removed<=counts).all())
            self.assertTrue((r.targets>=0).all() and (r.targets<=1).all())
            self.assertTrue(torch.isfinite(r.total))
            r.total.backward()
            self.assertTrue(torch.isfinite(weak.grad).all())
            self.assertTrue(torch.isfinite(strong.grad).all())

    def test_double_precision_gradient_check(self):
        weak = torch.tensor([[[4.,0.],[0.,2.],[1.,0.]]],dtype=torch.float64,requires_grad=True)
        strong = torch.randn(1,3,2,dtype=torch.float64,requires_grad=True)
        counts = torch.tensor([[2,1]])
        self.assertTrue(torch.autograd.gradcheck(lambda w,s: shrink_loss(w,s,counts,.95).total,(weak,strong)))

    def test_input_validation(self):
        w = torch.zeros(1,2,2)
        for counts in (torch.tensor([[1.,1.]]),torch.tensor([[0,1]]),torch.tensor([[-1,3]])):
            with self.assertRaises(ValueError): shrink_loss(w,w,counts)
        for tau in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): shrink_loss(w,w,torch.tensor([[1,1]]),tau)
        with self.assertRaises(FloatingPointError):
            shrink_loss(w+float('nan'),w,torch.tensor([[1,1]]))

    def test_ahil_bounded_detached_weights(self):
        w = torch.randn(2,4,3,requires_grad=True)
        s = torch.randn_like(w,requires_grad=True)
        r = reference_loss(w,s,torch.tensor([[2,2,0],[1,1,2]]),'ahil')
        self.assertFalse(r.weights.requires_grad)
        self.assertTrue(((r.weights>=0)&(r.weights<=1)).all())
        r.total.backward()
        self.assertTrue(torch.isfinite(w.grad).all())


if __name__ == '__main__': unittest.main()
