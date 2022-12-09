import torch

class WeightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("Entered")
            w = module.weight.data
            norm = w.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, min=0.0, max=5.0)
            w *= (desired / (1e-8 + norm))
            module.weight.data = w











