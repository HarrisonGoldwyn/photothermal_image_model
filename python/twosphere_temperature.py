import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## Get physical constants from Scipy
import scipy.constants as con
## Some useful numerical constants
hbar = con.physical_constants['Planck constant over 2 pi in eV s'][0]
c = con.physical_constants['speed of light in vacuum'][0]*1e2 #cm/m
k = con.physical_constants['Boltzmann constant in eV/K'][0]

import scipy.integrate as int

import scipy.io as sio

import scipy.optimize as opt

import scipy.special as spl

misloc_pack_path = os.path.join(
    os.path.sep,
    'Users',
    'chair',
    'Documents',
    'Academia',
    'SuperRes',
    'Biteen_colab',
    'Mispolarization',
    'python',
#     'gitted',
    )

sys.path.append(misloc_pack_path)

import misloc_mispol_package as mmp
import misloc_mispol_package.calc.coupled_dipoles as cp
import misloc_mispol_package.optics.anal_foc_diff_fields as aff



bisph_c = lambda epsilon: (epsilon**(-2) - 1)**0.5

def zeta_1_of_eps(epsilon):
    """ With epsilon, half the unitless seperation defined by
            seperation = 2 radius_1 / epsilon
        The argument of the returned arcsinh is half the foci-foci
        distence (units of length) on the sphere radius.
        Working with unitless parameters means
            a/radius_1 = c
        and
            a/radius_2 = c/(radius_2/radius_1)
        """
    c = bisph_c(epsilon)

    return np.arcsinh(c)


def zeta_2_of_eps(epsilon, l2_on_l1):
    """ With epsilon, half the unitless seperation defined by
            seperation = 2 radius_1 / epsilon
        The argument of the returned arcsinh is half the foci-foci
        distence (units of length) on the sphere radius.
        Working with unitless parameters means
            a/radius_1 = c
        and
            a/radius_2 = c/(radius_2/radius_1)
        """
    c = bisph_c(epsilon)

    return -np.arcsinh(c/l2_on_l1)


def coeff_F_m(m, epsilon, l2_on_l1, q2_on_q1):
    """ Returns mth coefficient
        """
    zeta_1 = zeta_1_of_eps(epsilon)
    zeta_2 = zeta_2_of_eps(epsilon, l2_on_l1)
    c = bisph_c(epsilon)

    numer = -2*2**(1/2)*c*(
        q2_on_q1*np.exp(
            -(m+(1/2))*np.abs(zeta_2))*np.cosh((m+3/2)*zeta_1)
        +
        np.exp(
            -(m+(1/2))*np.abs(zeta_1))*np.cosh((m+3/2)*zeta_2)
        )
    denom = ((m+1)*np.sinh((m+(3/2))*(zeta_1-zeta_2)))

    f_m = (
        numer
        /
        denom
        )

    return f_m


def coeff_tildeF_m(m, epsilon, l2_on_l1, q2_on_q1):
    """ Returns mth coefficient
        """
    zeta_1 = zeta_1_of_eps(epsilon)
    zeta_2 = zeta_2_of_eps(epsilon, l2_on_l1)
    c = bisph_c(epsilon)

    numer = 2*2**(1/2)*c*(
        q2_on_q1*np.exp(
            -(m+(1/2))*np.abs(zeta_2))*np.sinh((m+3/2)*zeta_1)
        +
        np.exp(
            -(m+(1/2))*np.abs(zeta_1))*np.sinh((m+3/2)*zeta_2)
        )
    denom = ((m+1)*np.sinh((m+(3/2))*(zeta_1-zeta_2)))

    f_m = (
        numer
        /
        denom
        )

    return f_m


def coeff_G2_m(m, epsilon, l2_on_l1):
    """ Returns mth coefficient
        """
    zeta_1 = zeta_1_of_eps(epsilon)
    zeta_2 = zeta_2_of_eps(epsilon, l2_on_l1)

    g_m = (
        1 - (m/(m+1))*(
            (
                np.cosh((m+(3/2))*zeta_1)*np.sinh((m-(1/2))*zeta_2)
                -
                np.cosh((m+(3/2))*zeta_2)*np.sinh((m-(1/2))*zeta_1)
                )
            /
            np.sinh((m+(3/2))*(zeta_1-zeta_2))
            )
        )

    return g_m


def coeff_G1_m(m, epsilon, l2_on_l1):
    """ Returns mth coefficient
        """
    zeta_1 = zeta_1_of_eps(epsilon)
    zeta_2 = zeta_2_of_eps(epsilon, l2_on_l1)

    g_m = (
        1 - (m/(m+1))*(
            (
                np.cosh((m+(3/2)*zeta_1))*np.cosh((m-(1/2))*zeta_2)
                -
                np.cosh((m+(3/2)*zeta_2))*np.cosh((m-(1/2))*zeta_1)
                )
            /
            np.sinh((m+(3/2))*(zeta_1-zeta_2))
            )
        )

    return g_m


def coeff_tildeG2_m(m, epsilon, l2_on_l1):
    """ Returns mth coefficient
        """
    zeta_1 = zeta_1_of_eps(epsilon)
    zeta_2 = zeta_2_of_eps(epsilon, l2_on_l1)

    g_m = (
        1 - (m/(m+1))*(
            (
                -
                np.sinh((m+(3/2)*zeta_1))*np.sinh((m-(1/2))*zeta_2)
                +
                np.sinh((m+(3/2)*zeta_2))*np.sinh((m-(1/2))*zeta_1)
                )
            /
            np.sinh((m+(3/2))*(zeta_1-zeta_2))
            )
        )

    return g_m


def coeff_tildeG1_m(m, epsilon, l2_on_l1):
    """ Returns mth coefficient
        """
    zeta_1 = zeta_1_of_eps(epsilon)
    zeta_2 = zeta_2_of_eps(epsilon, l2_on_l1)

    g_m = (
        1 - (m/(m+1))*(
            (
                -
                np.cosh((m+(3/2)*zeta_1))*np.cosh((m-(1/2))*zeta_2)
                +
                np.cosh((m+(3/2)*zeta_2))*np.cosh((m-(1/2))*zeta_1)
                )
            /
            np.sinh((m+(3/2))*(zeta_1-zeta_2))
            )
        )

    return g_m

def vec_F(m, epsilon, l2_on_l1, q2_on_q1):
    """ Vector F coefficient for vector recursion realtions.
        """
    tf_m = coeff_tildeF_m(m, epsilon, l2_on_l1, q2_on_q1)
    f_m = coeff_F_m(m, epsilon, l2_on_l1, q2_on_q1)

    vec_f = np.array([
        [tf_m,],
        [f_m,],
        ])
    return vec_f

def vec_G(m, epsilon, l2_on_l1):
    """ Vector G coefficient for vector recursion realtions.
        """
    g2 = coeff_G2_m(m, epsilon, l2_on_l1)
    g1 = coeff_G1_m(m, epsilon, l2_on_l1)
    g2tilde = coeff_tildeG2_m(m, epsilon, l2_on_l1)
    g1tilde = coeff_tildeG1_m(m, epsilon, l2_on_l1)

    vec_g = np.array([
        [g1tilde, -(1 - g2tilde)],
        [-(1 - g1), g2],
        ])
    return vec_g


def est_B_0(m_inf, epsilon, l2_on_l1, q2_on_q1):
    """ Estimates (returns) first coefficient B_0 from the limit
            B_0 = -lim_{m->inf} H_m
        where
            H_m = F_m + G_m H_{m-1} + (1 - G_m) H_{m-2}
            H_1 = F_1 + G_1 F_0
            H_0 = F_0
        """
    zeta_1 = zeta_1_of_eps(epsilon)
    zeta_2 = zeta_2_of_eps(epsilon, l2_on_l1)

    F_0 = vec_F(0, epsilon, l2_on_l1, q2_on_q1)
    F_1 = vec_F(1, epsilon, l2_on_l1, q2_on_q1)

    G_1 = vec_G(1, epsilon, l2_on_l1)

    H_0 = F_0
    if m_inf == 0:
        return -H_0
    H_1 = F_1 + G_1@F_0
    if m_inf == 1:
        return -H_1

    ## Iteratre through m vales from 2 to m_inf
    for m in range(2, m_inf+1):

        if m==2:
            H_m1 = H_1
            H_m2 = H_0

        G_m = vec_G(m, epsilon, l2_on_l1)
        F_m = vec_F(m, epsilon, l2_on_l1, q2_on_q1)

        H_m = F_m + G_m@H_m1 + (np.identity(2) - G_m)@H_m2

        ## Define for next iteration
        H_m1 = H_m
        H_m2 = H_m1

    b_0 = -H_m
    return b_0


def series_B_m(m, B_0, epsilon, l2_on_l1, q2_on_q1):
    """ Givens a column vector of the first coefficients 'B_0',
        returns an array of shape (2, m+1) containing the B^(1) and
        B^(2) coefficients from m=0 to m
        """
#     Bs = np.zeros((m+1, 2))
    Bs = np.zeros((2, m+1))
    Bs[:, [0]] = B_0

    if m==0:
        return Bs

    F_0 = vec_F(0, epsilon, l2_on_l1, q2_on_q1)
    B_1 = F_0 + B_0
    Bs[:, [1]] = B_1
    if m==1:
        return Bs

    ## Iteratre through m vales from 2 to m_inf
    for _m in range(2, m+1):

        B_m1 = Bs[:, [_m-1]]
        B_m2 = Bs[:, [_m-2]]

        G_m1 = vec_G(_m-1, epsilon, l2_on_l1)
        F_m1 = vec_F(_m-1, epsilon, l2_on_l1, q2_on_q1)

        Bs[:, [_m]] = F_m1 + G_m1@B_m1 + (np.identity(2) - G_m1)@B_m2

    return Bs


def temp_unitless_biph_coords(
    zeta,
    beta,
    l,
    epsilon,
    max_m,
    B_0,
    l2_on_l1,
    q2_on_q1,
    return_terms=False
    ):
    """ Unitless temperature around 2 spheres of radii 'l' and
        center-center seperation  as a function of bispherical
        coordinates '2l/epsilon'. The expression is defined as a series
        of coefficients B_m which are defined by recursion relation over

        Assuming zeta and betas correspond to coordinate pairs
        """
    zeta = np.asarray(zeta)
    beta = np.asarray(beta)

    ## Get array of B_m coefficients, shape (2, max_m+1)
    array_B_m = series_B_m(
        max_m,
        B_0,
        epsilon,
        l2_on_l1,
        q2_on_q1,
        )

    m_array = np.arange(0, max_m+1, 1)

    ## zetas and betas are coordinate pairs, so only one coordinate
    ## dimension is neccesary.
    legendre_P = np.zeros((len(beta), len(m_array)))

    for m in m_array:
        legendre_P[:, m] = spl.legendre(m)(np.cos(beta))

    sum_on_m_terms = (
        (
            array_B_m[0, None, :]
            *
            np.sinh((m_array[None, :]+0.5)*zeta[:, None])
            +
            array_B_m[1, None, :]
            *
            np.cosh((m_array[None, :]+0.5)*zeta[:, None])
            )
        *
        legendre_P
        )

    prefactor = np.sqrt(
        np.cosh(zeta)
        -
        np.cos(beta)
        )

    if return_terms:
        return prefactor[:, None]*sum_on_m_terms

    sum_on_m = np.sum(sum_on_m_terms, axis=-1)

    theta = (
        prefactor
        *
        sum_on_m
        )

    return theta


def biph_transform(x, y, z, epsilon):

    x = np.asarray(x)[:, None, None]
    y = np.asarray(y)[None, :, None]
    z = np.asarray(z)[None, None, :]

    c = np.sqrt(epsilon**(-.2)-1)
    R = np.sqrt(x**2. + y**2. + z**2.)
    Q = np.sqrt((R**2. + c**2.)**2. - (2*c*z)**2.)

    zeta = np.arcsinh(2*c*z/Q).ravel()
    beta = np.arccos((R**2. - c**2.)/Q).ravel()

    return (zeta, beta)

def hetero_sph_unitless_temp(
    which_sphere,
    epsilon,
    max_m,
    B_0,
    l2_on_l1,
    q2_on_q1,
    return_terms=False
    ):
    """ Returns analytic result for the average surface temperature in
        terms of another seires of the Bm coefficients
        """
    if which_sphere is 1:
        zeta_0 = zeta_1_of_eps(epsilon)
        _r = 1
    elif which_sphere is 2:
        zeta_0 = zeta_2_of_eps(epsilon, l2_on_l1)
        _r = l2_on_l1
    else:
        raise ValueError("'which_sphere' arg must be 1 or 2.")

    ## Get array of B_m coefficients, shape (2, max_m+1)
    array_B_m = series_B_m(
        max_m,
        B_0,
        epsilon,
        l2_on_l1,
        q2_on_q1,
        )

    m_array = np.arange(0, max_m+1, 1)

    exp_factor = np.zeros(m_array.shape)
    for m in m_array:
        exp_factor[m] = np.exp(-(m+1/2)*np.abs(zeta_0))

    sum_on_m_terms = (
        (
            array_B_m[0, :]
            *
            np.sinh(
                (m_array+0.5)*zeta_0
                )
            +
            array_B_m[1, :]
            *
            np.cosh(
                (m_array+0.5)*zeta_0
                )
            )
        *
        exp_factor
        )

    area = 4*np.pi*_r**2
    prefactor = (
        2 * np.pi * bisph_c(epsilon)**2 * 2 * np.sqrt(2)
        /
        (area*np.sinh(np.abs(zeta_0)))
        )

    if return_terms:
        return prefactor*sum_on_m_terms

    sum_on_m = np.sum(sum_on_m_terms, axis=-1)

    theta_bar = (
        prefactor
        *
        sum_on_m
        )

    return theta_bar


def converged_B_0(epsilon, m_inf, l2_on_l1, q2_on_q1, tol=.01):
    """ Iterate m->inf until B0 changes by less then 'tol' of last
        value """

    B0 = est_B_0(0, epsilon, l2_on_l1, q2_on_q1)

    for m in range(1, m_inf+1):

        new_B0 = est_B_0(m, epsilon, l2_on_l1, q2_on_q1)
        diff = new_B0 - B0

        if np.all(diff/new_B0 < tol):
            return new_B0

        else: B0 = new_B0

    raise ValueError("B_0 did not converge")
