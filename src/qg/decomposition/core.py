"""Core function to build a decomposition."""

from typing import Any

from qg.decomposition.base import SpaceTimeDecomposition
from qg.decomposition.exp_exp.core import GaussianExpBasis
from qg.decomposition.supports.space.base import SpaceSupportFunction
from qg.decomposition.supports.time.base import TimeSupportFunction
from qg.decomposition.taylor.core import TaylorFullFieldBasis
from qg.decomposition.taylor_exp.core import TaylorExpBasis
from qg.decomposition.wavelets.core import WaveletBasis

Basis = SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]

BASES: dict[str, type[Basis]] = {
    WaveletBasis.type: WaveletBasis,
    TaylorExpBasis.type: TaylorExpBasis,
    TaylorFullFieldBasis.type: TaylorFullFieldBasis,
    GaussianExpBasis.type: GaussianExpBasis,
}


def build_basis_from_params_dict(params: dict[str, Any]) -> Basis:
    """Build the right basis given params.

    Args:
        params (dict[str, Any]): Basis parameters (output of Basis.get_params).

    Returns:
        Basis: Corresponding basis.
    """
    return BASES[params["type"]].from_params(params)
