from .int4 import Q4Weight
import torch

def test_quant():
    weight = torch.randn(1024, 1024)
    qw = Q4Weight(weight)
    dequant_w = qw.dequant()
    diff = torch.mean(torch.abs(weight - dequant_w))
    assert diff < torch.std(weight) * 0.1
