"""
Linear and WENO reconstructions.
Louis Thiry, 2023
"""

import torch


def linear2(qm, qp):
    """
    2-points linear reconstruction:

    qm--x--qp

    """
    return 0.5 * (qm + qp)


def linear3_left(qm, q0, qp):
    """
    3-points linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    """
    return -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp


def linear4(qmm, qm, qp, qpp):
    """
    4-points linear reconstruction:

    qmm-----qm--x--qp-----qpp

    """
    return -1.0 / 12.0 * qmm + 7.0 / 12.0 * qm + 7.0 / 12.0 * qp - 1.0 / 12.0 * qpp


def linear5_left(qmm, qm, q0, qp, qpp):
    """
    5-points linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    """
    return (
        1.0 / 30.0 * qmm
        - 13.0 / 60.0 * qm
        + 47.0 / 60.0 * q0
        + 9.0 / 20.0 * qp
        - 1 / 20 * qpp
    )


def weno3(qm, q0, qp):
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """
    eps = 1e-8

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    beta1 = (q0 - qm) ** 2
    beta2 = (qp - q0) ** 2

    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    w1 = g1 / (beta1 + eps) ** 2
    w2 = g2 / (beta2 + eps) ** 2

    qi_weno3 = (w1 * qi1 + w2 * qi2) / (w1 + w2)

    return qi_weno3


def weno3z(qm, q0, qp):
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008).
    """
    eps = 1e-14

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    beta1 = (q0 - qm) ** 2
    beta2 = (qp - q0) ** 2
    tau = torch.abs(beta2 - beta1)

    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    w1 = g1 * (1.0 + tau / (beta1 + eps))
    w2 = g2 * (1.0 + tau / (beta2 + eps))

    qi_weno3 = (w1 * qi1 + w2 * qi2) / (w1 + w2)

    return qi_weno3


def weno5(qmm, qm, q0, qp, qpp):
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """
    eps = 1e-8
    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
    beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 / (beta1 + eps) ** 2
    w2 = g2 / (beta2 + eps) ** 2
    w3 = g3 / (beta3 + eps) ** 2

    qi_weno5 = (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)

    return qi_weno5


def weno5z(qmm, qm, q0, qp, qpp):
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """
    eps = 1e-16

    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
    beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2

    tau5 = torch.abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 * (1 + tau5 / (beta1 + eps))
    w2 = g2 * (1 + tau5 / (beta2 + eps))
    w3 = g3 * (1 + tau5 / (beta3 + eps))

    qi_weno5 = (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)

    return qi_weno5
