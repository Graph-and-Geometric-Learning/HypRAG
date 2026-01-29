import torch
import torch.nn as nn
from hyplib.manifolds import PoincareBall

class PoincareMLR(nn.Module):
    def __init__(self, manifold, c):
        """
        The Poincare multinomial logistic regression (MLR) operation.

        Parameters
        ----------
        x : tensor
            contains input values
        z : tensor
            contains the hyperbolic vectors describing the hyperplane orientations
        r : tensor
            contains the hyperplane offsets
        c : tensor
            curvature of the Poincare disk

        Returns
        -------
        tensor
            signed distances of input w.r.t. the hyperplanes, denoted by v_k(x) in
            the HNN++ paper
        """
        super(PoincareMLR, self).__init__()
        self.manifold = manifold
        self.c = c

    def forward(self, x, z, r):
        c_sqrt = self.c.sqrt()
        lam = 2 * (1 - self.c * x.pow(2).sum(dim=-1, keepdim=True))
        z_norm = z.norm(dim=0).clamp_min(1e-15)

        # Computation can be simplified if there is no offset
        if r is not None:
            two_csqrt_r = 2.0 * c_sqrt * r
            return (
                2
                * z_norm
                / c_sqrt
                * torch.asinh(
                    c_sqrt * lam / z_norm * torch.matmul(x, z) * two_csqrt_r.cosh()
                    - (lam - 1) * two_csqrt_r.sinh()
                )
            )
        else:
            return (
                2
                * z_norm
                / c_sqrt
                * torch.asinh(c_sqrt * lam / z_norm * torch.matmul(x, z))
            )