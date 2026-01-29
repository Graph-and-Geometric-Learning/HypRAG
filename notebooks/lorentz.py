from __future__ import annotations


import torch
import torch.nn as nn
import geoopt
from geoopt import Manifold
from geoopt import Lorentz as LorentzOri
from geoopt.utils import size2shape
import lmath as math


import itertools
from typing import Tuple, Any, Union, List, Optional
import functools
import operator

max_norm = 85
eps = 1e-8

__all__ = [
    "copy_or_set_",
    "strip_tuple",
    "size2shape",
    "make_tuple",
    "broadcast_shapes",
    "ismanifold",
    "canonical_manifold",
    "list_range",
    "idx2sign",
    "drop_dims",
    "canonical_dims",
    "sign",
    "prod",
    "clamp_abs",
    "sabs",
]


def copy_or_set_(dest: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """
    Copy or inplace set from :code:`source` to :code:`dest`.

    A workaround to respect strides of :code:`dest` when copying :code:`source`.
    The original issue was raised `here <https://github.com/geoopt/geoopt/issues/70>`_
    when working with matrix manifolds. Inplace set operation is mode efficient,
    but the resulting storage might be incompatible after. To avoid the issue we refer to
    the safe option and use :code:`copy_` if strides do not match.

    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor

    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


def strip_tuple(tup: Tuple) -> Union[Tuple, Any]:
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def make_tuple(obj: Union[Tuple, List, Any]) -> Tuple:
    if isinstance(obj, list):
        obj = tuple(obj)
    if not isinstance(obj, tuple):
        return (obj,)
    else:
        return obj


def prod(items):
    return functools.reduce(operator.mul, items, 1)


def sign(x):
    return torch.sign(x.sign() + 0.5)


def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)


def clamp_abs(x, eps: float = 1e-15):
    s = sign(x)
    return s * sabs(x, eps=eps)


def idx2sign(idx: int, dim: int, neg: bool = True):
    """
    Unify idx to be negative or positive, that helps in cases of broadcasting.

    Parameters
    ----------
    idx : int
        current index
    dim : int
        maximum dimension
    neg : bool
        indicate we need negative index

    Returns
    -------
    int
    """
    if neg:
        if idx < 0:
            return idx
        else:
            return (idx + 1) % -(dim + 1)
    else:
        return idx % dim


def drop_dims(tensor: torch.Tensor, dims: List[int]):
    # Workaround to drop several dims in :func:`torch.squeeze`.
    seen: int = 0
    for d in dims:
        tensor = tensor.squeeze(d - seen)
        seen += 1
    return tensor


def list_range(end: int):
    res: List[int] = []
    for d in range(end):
        res.append(d)
    return res


def canonical_dims(dims: List[int], maxdim: int):
    result: List[int] = []
    for idx in dims:
        result.append(idx2sign(idx, maxdim, neg=False))
    return result


def size2shape(*size: Union[Tuple[int], int]) -> Tuple[int]:
    return make_tuple(strip_tuple(size))


def broadcast_shapes(*shapes: Tuple[int]) -> Tuple[int]:
    """Apply numpy broadcasting rules to shapes."""
    result = []
    for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
        dim: int = 1
        for d in dims:
            if dim != 1 and d != 1 and d != dim:
                raise ValueError("Shapes can't be broadcasted")
            elif d > dim:
                dim = d
        result.append(dim)
    return tuple(reversed(result))


def ismanifold(instance, cls):
    """
    Check if interface of an instance is compatible with given class.

    Parameters
    ----------
    instance : geoopt.Manifold
        check if a given manifold is compatible with cls API
    cls : type
        manifold type

    Returns
    -------
    bool
        comparison result
    """
    if not issubclass(cls, geoopt.manifolds.Manifold):
        raise TypeError(
            "`cls` should be a subclass of geoopt.manifolds.Manifold")
    if not isinstance(instance, geoopt.manifolds.Manifold):
        return False
    else:
        # this is the case to care about, Scaled class is a proxy, but fails instance checks
        while isinstance(instance, geoopt.Scaled):
            instance = instance.base
        return isinstance(instance, cls)


def canonical_manifold(manifold: "geoopt.Manifold"):
    """
    Get a canonical manifold.

    If a manifold is wrapped with Scaled. Some attributes may not be available. This should help if you really need them.

    Parameters
    ----------
    manifold : geoopt.Manifold

    Returns
    -------
    geoopt.Maniflold
        an unwrapped manifold
    """
    while isinstance(manifold, geoopt.Scaled):
        manifold = manifold.base
    return manifold


def cosh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.cosh(x)


def sinh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.sinh(x)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=1e-9)  # Smaller epsilon due to precision around x=0.
    return torch.sqrt(x)


class LeakyClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        with torch.no_grad():
            ctx.save_for_backward(x.ge(min) & x.le(max))
            return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None


def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)


class Atanh(torch.autograd.Function):
    """
    Numerically stable arctanh that never returns NaNs.
    x = clamp(x, min=-1+eps, max=1-eps)
    Returns atanh(x) = arctanh(x) = 0.5*(log(1+x)-log(1-x)).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        x = clamp(x, min=-1. + 4 * eps, max=1. - 4 * eps)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        return grad_output / (1 - x**2)


def atanh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable arctanh that never returns NaNs.

    :param x: The input tensor.
    :return: log(x + sqrt(max(x^2 - 1, eps))
    """
    return Atanh.apply(x)


class Acosh(torch.autograd.Function):
    """
    Numerically stable arccosh that never returns NaNs.
    Returns acosh(x) = arccosh(x) = log(x + sqrt(max(x^2 - 1, eps))).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = clamp(x, min=1 + eps)
            z = sqrt(x * x - 1.)
            ctx.save_for_backward(z)
            return torch.log(x + z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        z, = ctx.saved_tensors
        # z_ = clamp(z, min=eps)
        z_ = z
        return grad_output / z_


def acosh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable arccosh that never returns NaNs.

    :param x: The input tensor.
    :return: log(x + sqrt(max(x^2 - 1, eps))
    """
    return Acosh.apply(x)


class Lorentz(LorentzOri):
    def __init__(self, k=1.0, learnable=False):
        """
        Initialize a Lorentz manifold with curvature k.

        Parameters:
            k (float): Curvature of the manifold.
            learnable (bool): If True, k is learnable. Default is False.
        """
        super().__init__(k, learnable)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[bool, Optional[str]]:
        """
        Check if a point lies on the manifold.

        Parameters:
            x (torch.Tensor): Point to check.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            dim (int): Dimension to check.

        Returns:
            Tuple[bool, Optional[str]]: A boolean indicating if the point is on the manifold, and an optional reason string.
        """
        dn = x.size(dim) - 1
        # print(x.shape)
        x = x ** 2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(dim=dim, keepdim=True)
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        reason = None if ok else f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        if not ok:
            print(quad_form)
            indices, _ = torch.where(torch.abs(quad_form + self.k) > atol)
            print(indices)
            print(x[indices])
            print(quad_form[indices])
        return ok, reason

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[
        bool, Optional[str]]:
        """
        Check if a vector lies on the tangent space at a point.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): Vector to check.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            dim (int): Dimension to check.

        Returns:
            Tuple[bool, Optional[str]]: A boolean indicating if the vector is on the tangent space, and an optional reason string.
        """
        inner_ = math.inner(u, x, dim=dim)
        ok = torch.allclose(inner_, torch.zeros(1), atol=atol, rtol=rtol)
        reason = None if ok else "Minkowski inner product is not equal to zero"
        return ok, reason

    def add_time(self, space):
        """ Concatenates time component to given space component. """
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        """ Calculates time component from given space component. """
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2 + self.k)

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        """
        Compute the distance between two points on the manifold.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.
            keepdim (bool): If True, retains the last dimension.
            dim (int): Dimension to compute distance.

        Returns:
            torch.Tensor: Distance between x and y.
        """
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        """
        Compute the distance from the origin to a point on the manifold.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            keepdim (bool): If True, retains the last dimension.
            dim (int): Dimension to compute distance.

        Returns:
            torch.Tensor: Distance from the origin to x.
        """
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    def cdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise geodestic distance between points in x and y.

        Parameters:
            x (torch.Tensor): First set of points.
            y (torch.Tensor): Second set of points.

        Returns:
            torch.Tensor: Pairwise distances between points in x and y.
        """
        return math.cdist(x, y, k=self.k)

    def lorentz_to_klein(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a point from Lorentz to Klein coordinates.

        Parameters:
            x (torch.Tensor): Point in Lorentz coordinates.

        Returns:
            torch.Tensor: Point in Klein coordinates.
        """
        dim = x.shape[-1] - 1
        return acosh(x.narrow(-1, 1, dim) / x.narrow(-1, 0, 1))

    def klein_to_lorentz(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a point from Klein to Lorentz coordinates.

        Parameters:
            x (torch.Tensor): Point in Klein coordinates.

        Returns:
            torch.Tensor: Point in Lorentz coordinates.
        """
        norm = (x * x).sum(dim=-1, keepdim=True)
        size = x.shape[:-1] + (1,)
        return torch.cat([x.new_ones(size), x], dim=-1) / torch.clamp_min(torch.sqrt(1 - norm), 1e-7)

    def lorentz_to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a point from Lorentz to Poincare coordinates.

        Parameters:
            x (torch.Tensor): Point in Lorentz coordinates.

        Returns:
            torch.Tensor: Point in Poincare coordinates.
        """
        return math.lorentz_to_poincare(x, self.k)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        """
        Compute the norm of a tangent vector.

        Parameters:
            u (torch.Tensor): Tangent vector.
            keepdim (bool): If True, retains the last dimension.
            dim (int): Dimension to compute the norm.

        Returns:
            torch.Tensor: Norm of u.
        """
        return math.norm(u, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Convert Euclidean gradient to Riemannian gradient.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): Euclidean gradient.
            dim (int): Dimension to compute the gradient.

        Returns:
            torch.Tensor: Riemannian gradient.
        """
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Project a point onto the manifold.

        Parameters:
            x (torch.Tensor): Point to project.
            dim (int): Dimension to project.

        Returns:
            torch.Tensor: Projected point.
        """
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Project a tangent vector onto the tangent space at a point.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            v (torch.Tensor): Tangent vector to project.
            dim (int): Dimension to project.

        Returns:
            torch.Tensor: Projected tangent vector.
        """
        v = math.project_u(x, v, k=self.k, dim=dim)
        return v

    def proju0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project a tangent vector onto the tangent space at the origin.

        Parameters:
            v (torch.Tensor): Tangent vector to project.

        Returns:
            torch.Tensor: Projected tangent vector.
        """
        return math.project_u0(v)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1) -> torch.Tensor:
        """
        Perform the exponential map to move from a point in the tangent space to the manifold.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): Tangent vector.
            norm_tan (bool): If True, normalize the tangent vector. Default is True.
            project (bool): If True, project the result back onto the manifold. Default is True.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Point on the manifold.
        """
        if norm_tan:
            u = self.proju(x, u, dim=dim)
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        """
        Perform the exponential map from the origin.

        Parameters:
            u (torch.Tensor): Tangent vector.
            project (bool): If True, project the result back onto the manifold. Default is True.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Point on the manifold.
        """
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform the logarithmic map to move from a point on the manifold to the tangent space.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            y (torch.Tensor): Point on the manifold.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Tangent vector.
        """
        return math.logmap(x, y, k=self.k, dim=dim)

    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform the logarithmic map from the origin.

        Parameters:
            y (torch.Tensor): Point on the manifold.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Tangent vector.
        """
        return math.logmap0(y, k=self.k, dim=dim)

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform the inverse logarithmic map to move from the tangent space to the manifold.

        Parameters:
            x (torch.Tensor): Tangent vector.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Point on the manifold.
        """
        return math.logmap0back(x, k=self.k, dim=dim)

    def inner(self, u: torch.Tensor, v: Optional[torch.Tensor] = None, *, keepdim=False,
              dim=-1) -> torch.Tensor:
        """
        Compute the inner product of two tangent vectors at a point.

        Parameters:
            u (torch.Tensor): First tangent vector.
            v (torch.Tensor, optional): Second tangent vector. If None, uses u.
            keepdim (bool): If True, retains the last dimension. Default is False.
            dim (int): Dimension to compute the inner product.

        Returns:
            torch.Tensor: Inner product.
        """
        if v is None:
            v = u
        return math.inner(u, v, dim=dim, keepdim=keepdim)

    def inner0(self, v: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        """
        Compute the inner product of a tangent vector at the origin.

        Parameters:
            v (torch.Tensor): Tangent vector.
            keepdim (bool): If True, retains the last dimension. Default is False.
            dim (int): Dimension to compute the inner product.

        Returns:
            torch.Tensor: Inner product.
        """
        return math.inner0(v, k=self.k, dim=dim, keepdim=keepdim)

    def cinner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-inner product of two points.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Cross-inner product.
        """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform parallel transport of a tangent vector.

        Parameters:
            x (torch.Tensor): Starting point on the manifold.
            y (torch.Tensor): Ending point on the manifold.
            v (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform parallel transport from the origin.

        Parameters:
            y (torch.Tensor): Ending point on the manifold.
            u (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform inverse parallel transport to the origin.

        Parameters:
            x (torch.Tensor): Starting point on the manifold.
            u (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        return math.parallel_transport0back(x, u, k=self.k, dim=dim)

    def transp_follow_expmap(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1,
                             project=True) -> torch.Tensor:
        """
        Perform parallel transport following an exponential map.

        Parameters:
            x (torch.Tensor): Starting point on the manifold.
            u (torch.Tensor): Tangent vector for exponential map.
            v (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.
            project (bool): If True, project the result back onto the manifold. Default is True.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform Mobius addition of two points.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Result of Mobius addition.
        """
        v = self.logmap0(y)
        v = self.transp0(x, v)
        return self.expmap(x, v)

    def geodesic_unit(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True) -> torch.Tensor:
        """
        Compute a point on a geodesic given a time parameter.

        Parameters:
            t (torch.Tensor): Time parameter.
            x (torch.Tensor): Starting point on the manifold.
            u (torch.Tensor): Tangent vector.
            dim (int): Dimension to perform the operation.
            project (bool): If True, project the result back onto the manifold. Default is True.

        Returns:
            torch.Tensor: Point on the geodesic.
        """
        res = math.geodesic_unit(t, x, u, k=self.k)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None) -> geoopt.ManifoldTensor:
        """
        Create a random point on the manifold, induced by a normal distribution on the tangent space of zero.

        Parameters:
            size: Desired shape.
            mean (float or torch.Tensor): Mean value for the normal distribution.
            std (float or torch.Tensor): Standard deviation value for the normal distribution.
            dtype (torch.dtype): Target dtype for the sample. Should match manifold dtype if not None.
            device (torch.device): Target device for the sample. Should match manifold device if not None.

        Returns:
            geoopt.ManifoldTensor: Random points on the hyperboloid.
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError("`device` does not match the projector `device`, set the `device` argument to None")
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError("`dtype` does not match the projector `dtype`, set the `dtype` argument to None")
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        # tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    def origin(self, *size, dtype=None, device=None, seed=42) -> geoopt.ManifoldTensor:
        """
        Create a zero point origin on the manifold.

        Parameters:
            size: Desired shape.
            dtype (torch.dtype): Desired dtype.
            device (torch.device): Desired device.
            seed (int): Ignored.

        Returns:
            geoopt.ManifoldTensor: Zero point on the manifold.
        """
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k)
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    def mid_point(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the midpoint of points on the manifold.

        Parameters:
            x (torch.Tensor): Points on the manifold.
            w (torch.Tensor, optional): Weights for each point. Default is None.

        Returns:
            torch.Tensor: Midpoint.
        """
        if w is not None:
            ave = w.matmul(x)
        else:
            ave = x.mean(dim=-2)
        denom = (-math.inner(ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt()
        return self.k.sqrt() * ave / denom

    def squared_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared distance between two points on the manifold.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Squared distance between x and y.
        """
        return -2 * self.k - 2 * math.inner(x, y, keepdim=True)

    def geodesic_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic distance between two points on the manifold.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Geodesic distance between x and y.
        """
        return math.arcosh(-math.inner(x, y) / self.k) * self.k.sqrt()

    def pairwise_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise inner product between points in x and y.

        Parameters:
            x (torch.Tensor): First set of points.
            y (torch.Tensor): Second set of points.

        Returns:
            torch.Tensor: Pairwise inner products between points in x and y.
        """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def pairwise_squared_dist(self, x: torch.Tensor, y: torch.Tensor, detach=False) -> torch.Tensor:
        """
        Compute the pairwise squared distance between points in x and y.

        Parameters:
            x (torch.Tensor): First set of points.
            y (torch.Tensor): Second set of points.

        Returns:
            torch.Tensor: Pairwise squared distances between points in x and y.
        """
        if detach:
            x = x.detach()
            y = y.detach()
            k = self.k.detach()
        else:
            k = self.k

        k = k.to(x.device)

        return -2 * k - 2 * self.pairwise_inner(x, y)

    def pairwise_geodesic_dist(self, x: torch.Tensor, y: torch.Tensor, detach=False) -> torch.Tensor:
        """
        Compute the pairwise geodesic distance between points in x and y.

        Parameters:
            x (torch.Tensor): First set of points.
            y (torch.Tensor): Second set of points.

        Returns:
            torch.Tensor: Pairwise geodesic distances between points in x and y.
        """
        if detach:
            x = x.detach()
            y = y.detach()
            k = self.k.detach()
        else:
            k = self.k

        k = k.to(x.device)

        return math.arcosh(-self.pairwise_inner(x, y) / k) * k.sqrt()

    def oxy_angle(self, x: torch.Tensor, y: torch.Tensor, eps=1e-7) -> torch.Tensor:
        """
        Given two vectors `x` and `y` on the hyperboloid, compute the exterior
        angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
        of the hyperboloid.

        This expression is derived using the Hyperbolic law of cosines.

        Args:
            x: Tensor of shape `(B, D)` giving a batch of space components of
                vectors on the hyperboloid.
            y: Tensor of same shape as `x` giving another batch of vectors.

        Returns:
            Tensor of shape `(B, )` giving the required angle. Values of this
            tensor lie in `(0, pi)`.
        """

        # Calculate time components of inputs (multiplied with `sqrt(curv)`):
        # assert x.shape == y.shape
        # x_om = self._check_point_on_manifold(x.detach())
        # assert x_om[0], x_om[1]
        # y_om = self._check_point_on_manifold(y.detach())
        # assert y_om[0], y_om[1]

        # Calculate lorentzian inner product multiplied with curvature. We do not use
        # the `pairwise_inner` implementation to save some operations (since we only
        # need the diagonal elements).
        # c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
        c_xyl = 1/self.k * math.inner(x, y)

        # Make the numerator and denominator for input to arc-cosh, shape: (B, )
        # acos_numer = y_time + c_xyl * x_time
        acos_numer = y[..., 0] + c_xyl * x[..., 0]
        # acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
        acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

        # acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
        acos_input = acos_numer / (torch.norm(x[..., 1:], dim=-1) * acos_denom + eps)
        _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

        return _angle

    def half_aperture(self, x: torch.Tensor, min_radius: float = 0.1, eps: float = 1e-7) -> torch.Tensor:
        """
        Compute the half aperture angle of the entailment cone formed by vectors on
        the hyperboloid. The given vector would meet the apex of this cone, and the
        cone itself extends outwards to infinity.

        Args:
            x: Tensor of shape `(B, D)` giving a batch of space components of
                vectors on the hyperboloid.
            min_radius: Radius of a small neighborhood around vertex of the hyperboloid
                where cone aperture is left undefined. Input vectors lying inside this
                neighborhood (having smaller norm) will be projected on the boundary.
            eps: Small float number to avoid numerical instability.

        Returns:
            Tensor of shape `(B, )` giving the half-aperture of entailment cones
            formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
        """

        # Ensure numerical stability in arc-sin by clamping input.
        x_space = x[..., 1:]
        asin_input = 2 * min_radius / (torch.norm(x_space, dim=-1) / self.k.sqrt() + eps)
        _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

        return _half_aperture