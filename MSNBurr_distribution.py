import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable
from typing import Optional, Tuple
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none

def logp(y: TensorVariable, mu: TensorVariable, sigma: TensorVariable, alpha: TensorVariable):
    omega = (1 + (1 / alpha))**(alpha + 1) / np.sqrt(2 * np.pi)
    epart = -pt.log(alpha) - (omega / sigma * (y - mu))

    logpdf = pt.switch(
        pt.eq(y, -np.inf),
        -np.inf,
        pt.log(omega) - pt.log(sigma) - (omega / sigma * (y - mu)) - ((alpha + 1) * pt.log1pexp(epart))
    )
    return check_parameters(
        logpdf,
        alpha > 0,
        sigma > 0,
        msg=f"alpha must be more than 0, sigma must be more than 0"
    )

def logcdf(y: TensorVariable, mu: TensorVariable, sigma: TensorVariable, alpha: TensorVariable, **kwargs):
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    epart = -pt.log(alpha) - (omega/sigma*(y-mu))
    logcdf= -alpha*pt.log1pexp(epart)
    return check_parameters(
        logcdf,
        alpha > 0,
        sigma > 0,
        msg=f"alpha must more than 0, sigma must more than 0",
    )

def moment(rv, size, mu: TensorVariable, sigma: TensorVariable, alpha: TensorVariable):
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    moment= mu + sigma/omega*(pt.digamma(alpha)-pt.digamma(1)-pt.log(alpha))
    if not rv_size_is_none(size):
        moment = pt.full(size, moment)
    return moment

def random(
      mu: np.ndarray | float,
      sigma: np.ndarray | float,
      alpha: np.ndarray | float,
      rng = np.random.default_rng(),
      size: Optional[Tuple[int]]=None,
    ):
    if sigma <= 0:
        raise ValueError("sigma must more than 0")
    if alpha <= 0:
        raise ValueError("alpha must more than 0")
    u = rng.uniform(low=0, high=1, size=size)
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    random = mu - sigma/omega*(np.log(alpha)+np.log((u**(-1/alpha))-1))
    return np.asarray(random)

class msnburr:
    def __new__(self, name:str, mu, sigma, alpha, observed=None, **kwargs):
        return pm.CustomDist(
            name,
            mu, sigma, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
            moment=moment,
            observed=observed,
            **kwargs
        )

    @classmethod
    def dist(cls, mu, sigma, alpha, **kwargs):
        return pm.CustomDist.dist(
            mu, sigma, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
            moment=moment
        )
