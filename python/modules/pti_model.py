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



probe_field_magnitude = (1.4444)**0.5 * 10**18 * 10**(-1)


class heated_particle_properties(object):

    def __init__(self,
        hw,
        eps_inf,
        w_p,
        gamma,
        eps0,
        E_probe=probe_field_magnitude,
        dn_dt=None,
        dielectric_model='Drude',
        **kwargs):
        """ Generates dielectric funtion at given wavelength and Drude parameters
            for use by the methods """
        # eps1 = -2.5676 + 1j*3.6391

        if dielectric_model == 'Drude':
            self.eps1 = cp.drude_model(
                w=hw/hbar,
                eps_inf=eps_inf,  # eps_inf
                w_p=w_p,  # w_p
                gamma=gamma,)
        elif dielectric_model == 'DL':
            self.eps1 = cp.drude_lorentz_model(
                w=hw/hbar,
                eps_inf=eps_inf,  # eps_inf
                w_p=w_p,  # w_p
                gamma=gamma,
                **kwargs)

        self.eps0 = eps0
        self.n0 = eps0**0.5
        self.E_0 = E_probe

        ## Change in refractive index with temperature
        if dn_dt is None:
            ## Default is 5.858*10**(-4) if n0 is glyucerol (1.473)
            self.dn_dt = -2.7*10**(-4) * self.n0
        else:
            self.dn_dt = dn_dt

    def deltaNonN(self, T, eps0 = None):
        """ Define change in ref index with temperature
            """
        if eps0 is None:
            n0 = self.n0
        else:
            n0 = eps0**0.5

        return self.dn_dt/n0*T


    def alphaC(self, a, eps0=None):
        """ Define core polarizability of a sphere as
            funtion of core radius.
            """

        if eps0 is None:
            eps0 = self.eps0

        return a**3*((self.eps1 - eps0)/(self.eps1 + 2*eps0))


    def delT_shell_on_delT_core(self, b, a):
        """ Define the ratio of shell to core temperature in
            in terms of the core radius 'a' and the shell
            radius 'b'
            """
        if b == a:
            return 0

        ratio = (
            (-3 * a * (a**2. - b**2.))
            /
            (2*(-a**3. + b**3.))
            )

        return ratio


    def alphaDelta(self,
        T,
        b,
        a,
        eps0=None,
        lounis_expansion=True):
        """ Change in alpha with temperature of the glycerol shell"""

        if eps0 is None:
            eps0 = self.eps0

        if b == a:
            return 0

        f = a**3. / b**3.

        if not lounis_expansion:
            da = (
                b**3./3 * (1-f)
                *
                (1 + 2*f*(
                    (self.eps1 - eps0)/(self.eps1 + 2*eps0)
                    )**2.)
                *
                self.deltaNonN(T, eps0=eps0)
                *
                self.delT_shell_on_delT_core(b, a)
                )
        elif lounis_expansion:
            da = (
                - (2/3) * b**3. * (f-1)
                *
                (
                    1
                    +
                    2*f*(self.eps1 - eps0)**2.
                    /
                    (self.eps1 + 2*eps0)**2.
                    )
                *
                self.deltaNonN(T, eps0=eps0)
                *
                self.delT_shell_on_delT_core(b, a)
                )

        return da


    def alphaDeltaCN(self,
        T,
        a,
        eps0=None,
        lounis_expansion=True,
        dnr_dt=None,
        dni_dt=None,
        dnr_dt_fof_T=None,
        dni_dt_fof_T=None,):
        """ Change in polarizability due to change in core temp"""
        if dnr_dt is None:
            dnr_dt = - 2.7 * 10**(-4)
        if dni_dt is None:
            dni_dt = - 4 * 10**(-4)

        if eps0 is None:
            eps0 = self.eps0

        n_p = np.sqrt((np.abs(self.eps1) + np.real(self.eps1))/2)
        n_pp = np.sqrt((np.abs(self.eps1) - np.real(self.eps1))/2)

        if type(dnr_dt) is np.array:
            dnr_dt = dnr_dt[:, None]

        if type(dnr_dt) is np.array:
            dni_dt = dni_dt[:, None]
        da = (
            (6 * a**3 * eps0)
            /
            (self.eps1 + 2*eps0)**2
            *
            (n_p + 1j*n_pp)
            *
            (dnr_dt + 1j*dni_dt) * T
            )

        return da


    def alpha_of_T(self, T, b, a, **kwargs):

        alpha = (
            self.alphaC(a)
            +
            self.alphaDelta(T, b, a)
            +
            self.alphaDeltaCN(T, a, **kwargs)
            )

        return alpha


    def foc_field(self, dip_angle, x, y, k):
        """ Dipole field of an x oriented dipole"""

        E = aff.E_field(
            dipole_orientation_angle=dip_angle,
            xi=x,
            y=y,
            k=k
            )

        return E


class single_particle_image(heated_particle_properties):

    def __init__(self,
        hw,
        eps_inf,
        w_p,
        gamma,
        eps0,
        E_probe=probe_field_magnitude,
        dn_dt=None,
        metal_room_temp=False,
        **kwargs):

        heated_particle_properties.__init__(
            self,
            hw,
            eps_inf,
            w_p,
            gamma,
            eps0,
            E_probe,
            dn_dt,
            **kwargs
            )
        self.metal_room_temp = metal_room_temp

    def alphaDeltaCN(self, T, a, **kwargs):
        """ Change in polarizability due to change in core temp"""
        if not self.metal_room_temp:
            return heated_particle_properties.alphaDeltaCN(self, T, a, **kwargs)
        elif self.metal_room_temp:
            return 0


    def p(self,
        alpha,
        gScale=1,
        probe_E=None):
        if probe_E is None:
            probe_E = self.E_0

        _p = np.asarray([alpha*probe_E])

        if len(_p.ravel()) is 1:
            _p = _p.ravel()

        return _p


    def p1hotFoc(self,
        w, T1, b1, a1,
        gScale=1,
        **kwargs):

        alpha1 = self.alpha_of_T(T1, b1, a1, **kwargs)

        p1 = self.p(
            alpha=alpha1,
            gScale=gScale)

        return p1

    def p1coldFoc(self,
        w, T1, b1, a1,
        gScale=1):

        alpha1 = self.alphaC(a1)

        p1 = self.p(
            alpha=alpha1,
            gScale=gScale)

        return p1

    def hot_I(self,
        T1, w, l, b1, a1, x1,
    #     dip_angle_1=0,
    #     dip_angle_2=0,
        gScale=1,
        **kwargs):

    #     print(f'kwargs = {kwargs}')
        image = self.dip_image(
            T1, w, l, b1, a1, x1,
            pp1_func=self.p1hotFoc,
            **kwargs,
            )

        return image

    def cold_I(self,
        T1, w, l, b1, a1, x1,
        **kwargs):

        image = self.dip_image(
            T1, w, l, b1, a1, x1,
            pp1_func=self.p1coldFoc,
            **kwargs,
            )

        return image

    def dip_image(self,
        T1, w, l, b1, a1, x1,
        pp1_func,
        gScale=1,
        n_b=None,
        **kwargs
        ):

        if n_b==None:
            n_b=self.n0

        k = w*n_b/c

        x1 = l - x1

        def dot(field1, field2):
            return np.sum(field1 * field2, axis=0)

        field1 = self.foc_field(0, x1, 0, k)

        pp1 = pp1_func(
            w, T1, b1, a1,
            gScale=1
            **kwargs
            )

        t1 = np.abs(pp1)**2. * dot(field1, np.conj(field1))

        return t1

    def wfi_pti(self,
        T1, w, l, b1, a1, x1,
    #     dip_angle_1=0,
    #     dip_angle_2=0,
        **kwargs):
        """ Lengths are input in nm

            Args:
                T1: Temperature of particle (core)
                w: frequency of probe field
                l: x positions to evaluate signal
                b1: radius of shell surface of particle
                a1: radius of core of particle
                x1: position of single particle
            """

        ## Convert all length args back to cgs (in nm)
        l, b1, a1, x1 =  1e-7*np.array([l, b1, a1, x1])


        difference_I = (
            self.hot_I(
                T1, w, l, b1, a1, x1,
                **kwargs)
            -
            self.cold_I(
                T1, w, l, b1, a1, x1,
                **kwargs)
            )

        return difference_I

    ## Confocal image

    def p_gau_probed(self,
        x_beam,
        probe_hw,
        particle_position,
        alpha,
        gscale=1,
        probe_E=None,
        n_b=None):
        """ Define dipole magnitude with Gaussian beam driving
            force.
            """
        if probe_E is None:
            probe_E = self.E_0

        if n_b==None:
            n_b=self.n0

        x1 = np.array([particle_position, 0, 0])

        if type(alpha) is np.ndarray:
            alpha = np.asarray(alpha).reshape(len(alpha), 1, 1)

        p1 = cp.single_dip_mag_focused_beam(
            angle=0,
            p0_position=x1,
            beam_x_positions=x_beam,
            E_d_angle=0,
            drive_hbar_w=probe_hw,
            alpha0_diag=alpha*np.identity(3)[None, ...],
            n_b=n_b,
            drive_amp=probe_E,
            return_polarizabilities=False,)

        return p1


    def p1_hot_conf(self,
        x_beam,
        probe_hw, T1, b1, particle_position, a1,
        gscale=1,
        **kwargs):
        """ Confocally probed dipole 1"""

        alpha1 = self.alpha_of_T(T1, b1, a1, **kwargs)

        p1 = self.p_gau_probed(
            x_beam,
            probe_hw=probe_hw,
            particle_position=particle_position,
            alpha=alpha1,
            gscale=gscale)

        return p1


    def p1_cold_conf(self,
        x_beam,
        probe_hw, b1, particle_position, a1,
        gscale=1,
        **kwargs):

        return self.p1_hot_conf(
            x_beam,
            probe_hw,
            T1=0,
            b1=b1,
            particle_position=particle_position,
            a1=a1,
            gscale=gscale,
            **kwargs)


    def conf_PTI(self,
        x_beam,
        probe_hbar_w, T1, b1, particle_position, a1,
        **kwargs):
        """ Args are all simple floats. Lengths are in units of nm """

        x_beam, particle_position, b1, a1 =  1e-7*np.array(
            [x_beam, particle_position, b1, a1,])

        hot_dipole = lambda omega: self.p1_hot_conf(
            x_beam,
            probe_hw=omega*hbar,
            particle_position=particle_position,
            T1=T1,
            b1=b1,
            a1=a1,
            gscale=1,
            **kwargs)

        cold_dipole = lambda omega: self.p1_cold_conf(
            x_beam,
            probe_hw=omega*hbar,
            particle_position=particle_position,
            b1=b1,
            a1=a1,
            gscale=1,
            **kwargs)


        x1 = np.array([[particle_position, 0, 0]])
        # hot_signal = self.power_scatt_two_dips(
        #     p1_hot,
        #     p2_hot,
        #     x_beam,
        #     w, T1, b1, T2, b2, d, a1, a2, **kwargs)
        hot_signal = cp.single_dip_sigma_scat(
            hot_dipole,
            probe_hbar_w,
            n_b=self.n0,
            E_0=self.E_0,
            )

        cold_signal = cp.single_dip_sigma_scat(
            cold_dipole,
            probe_hbar_w,
            n_b=self.n0,
            E_0=self.E_0,
            )

        return hot_signal - cold_signal



class temperature_dependent_particle(heated_particle_properties):

    def __init__(self,
        hw,
        eps_inf,
        w_p,
        gamma,
        eps0,
        E_probe=probe_field_magnitude,
        dn_dt=None,
        **kwargs):

        heated_particle_properties.__init__(
            self,
            hw,
            eps_inf,
            w_p,
            gamma,
            eps0,
            E_probe,
            dn_dt,
            **kwargs)

    def g(
        self,
        d_angle,
        d, w,
        scale=1,
        n_b=None,
        p1_angle=0, p2_angle=0,):

        if n_b==None:
            n_b=self.n0
        ## rotate d vector
        R = cp.rotation_by(d_angle)
        d_col = (R @ np.array([[d, 0, 0]]).T).T

        ## Get coupling strength from misloc package
        G = cp.G(
            drive_hbar_w=hbar*w,
            d_col=d_col,
            n_b=n_b)
        ## but this was implemented to process multiple seperations d or
        ## frequencies w, so returns shape (1, 3, 3)
        if G.shape[0] is 1:
            G = G[0]

        def p_hat(angle):
            return cp.rotation_by(angle) @ np.array([1,0,0])[:,None]

        g = p_hat(p1_angle).T @ G @ p_hat(p2_angle)

        return g[...,0,:]


    def p_coupled( self,
        w, T1, b1, T2, b2, d, a1, a2,
        alpha, other_alpha,
        gScale=1,
        probe_E=None):
        if probe_E is None:
            probe_E = self.E_0

        p = (
            (alpha*(1 + self.g(0, d, w, gScale)*other_alpha)*probe_E)
            /
            (1 - alpha*other_alpha*self.g(0, d, w, gScale)**2)
            )

        if len(p.ravel()) is 1:
            p = p.ravel()

        return p


    def p1hotFoc(self,
        w, T1, b1, T2, b2, d, a1, a2,
        gScale=1):

        alpha1 = self.alpha_of_T(T1, b1, a1)

        alpha2 = self.alpha_of_T(T2, b2, a2)

        p1 = self.p_coupled(
            w, T1, b1, T2, b2, d, a1, a2,
            alpha=alpha1,
            other_alpha=alpha2,
            gScale=gScale)

        return p1


    def p2hotFoc(self,
        w, T1, b1, T2, b2, d, a1, a2,
        gScale=1):

        alpha1 = self.alpha_of_T(T1, b1, a1)

        alpha2 = self.alpha_of_T(T2, b2, a2)

        p2 = self.p_coupled(
            w, T1, b1, T2, b2, d, a1, a2,
            alpha=alpha2,
            other_alpha=alpha1,
            gScale=gScale)

        return p2


    def p1coldFoc(self,
        w, T1, b1, T2, b2, d, a1, a2,
        gScale=1):

        alpha1 = self.alphaC(a1)

        alpha2 = self.alphaC(a2)

        p1 = self.p_coupled(
            w, T1, b1, T2, b2, d, a1, a2,
            alpha=alpha1,
            other_alpha=alpha2,
            gScale=gScale)

        return p1


    def p2coldFoc(self,
        w, T1, b1, T2, b2, d, a1, a2,
        gScale=1):

        alpha1 = self.alphaC(a1)

        alpha2 = self.alphaC(a2)

        p2 = self.p_coupled(
            w, T1, b1, T2, b2, d, a1, a2,
            alpha=alpha2,
            other_alpha=alpha1,
            gScale=gScale)

        return p2


    def dip_image(self,
        T1, T2, w, l, d, b1, b2, a1, a2,
        pp1_func,
        pp2_func,
    #     dip_angle_1=0,
    #     dip_angle_2=0,
        gScale=1,
        n_b=None,
        return_components=False):

        if n_b==None:
            n_b=self.n0

        k = w*n_b/c
        x1 = l + d/2
        x2 = l - d/2


        def dot(field1, field2):
            return np.sum(field1 * field2, axis=0)

        field1 = self.foc_field(0, x1, 0, k)
        field2 = self.foc_field(0, x2, 0, k)

        pp1 = pp1_func(
            w, T1, b1, T2, b2, d, a1, a2,
            gScale=1
            )
        pp2 = pp2_func(
            w, T1, b1, T2, b2, d, a1, a2,
            gScale=1
            )

        t1 = np.abs(pp1)**2. * dot(field1, np.conj(field1))
        t2 = np.abs(pp2)**2. * dot(field2, np.conj(field2))
        t3 = (
            np.abs(pp1)*np.abs(pp2)
            *
            2*np.real(dot(field1, np.conj(field2)))
            *
            np.cos(
                np.arctan2(np.imag(pp1), np.real(pp1))
                -
                np.arctan2(np.imag(pp2), np.real(pp2))
                )
            )

        if not return_components:
            return t1 + t2 + t3
        elif return_components:
            return np.array([t1, t2, t3])
        else: print('what?...')


    def hot_I(self,
        T1, T2, w, l, d, b1, b2, a1, a2,
    #     dip_angle_1=0,
    #     dip_angle_2=0,
        gScale=1,
        **kwargs):

    #     print(f'kwargs = {kwargs}')
        image = self.dip_image(
            T1, T2, w, l, d, b1, b2, a1, a2,
            pp1_func=self.p1hotFoc,
            pp2_func=self.p2hotFoc,
            **kwargs,
    #         dip_angle_1=dip_angle_1,
    #         dip_angle_2=dip_angle_2,
            )

        return image


    def cold_I(self,
        T1, T2, w, l, d, b1, b2, a1, a2,
    #     dip_angle_1=0,
    #     dip_angle_2=0,
        **kwargs):

        image = self.dip_image(
            T1, T2, w, l, d, b1, b2, a1, a2,
            pp1_func=self.p1coldFoc,
            pp2_func=self.p2coldFoc,
            **kwargs,
    #         dip_angle_1=dip_angle_1,
    #         dip_angle_2=dip_angle_2,
            )

        return image


    def wfi_pti(self,
        T1, T2, w, l, d, b1, b2, a1, a2,
    #     dip_angle_1=0,
    #     dip_angle_2=0,
        **kwargs):
        """ Lengths are input in nm

            Args:
                T1: Temperature of particle 1 (core)
                T2: Temperature of particle 2 (core)
                w: frequency of probe field
                l: x positions to evaluate signal
                d: seperation between particle center points
                b1: radius of shell surface of particle 1
                b2: radius of shell surface of particle 2
                a1: radius of core of particle 1
                a2: radius of core of particle 2
            """

        ## Convert all length args back to cgs (in nm)
        l, d, b1, b2, a1, a2 =  1e-7*np.array([l, d, b1, b2, a1, a2])


        difference_I = (
            self.hot_I(
                T1, T2, w, l, d, b1, b2, a1, a2,
            #     dip_angle_1=0,
            #     dip_angle_2=0,
                **kwargs)
            -
            self.cold_I(
                T1, T2, w, l, d, b1, b2, a1, a2,
            #     dip_angle_1=0,
            #     dip_angle_2=0,
                **kwargs)
            )

        return difference_I



    ## Scan

    def p_gau_probed(self,
        x_beam,
        probe_hw,
        d,
        alpha, that_alpha,
        gscale=1,
        probe_E=None,
        n_b=None):
        """ Define dipole magnitude with Gaussian beam driving
            force.
            """
        if probe_E is None:
            probe_E = self.E_0

        if n_b==None:
            n_b=self.n0

        d_col_vec = np.array([[d, 0, 0]])

        if type(alpha) is np.ndarray:
            alpha = np.asarray(alpha).reshape(len(alpha), 1, 1)
            that_alpha = np.asarray(that_alpha).reshape(len(alpha), 1, 1)
        # else:
            # alpha = alpha
            # that_alpha = that_alpha


        p1, p2 = cp.coupled_dip_mags_focused_beam(
            mol_angle=0,
            plas_angle=0,
            d_col=-d_col_vec,
            p0_position=-d_col_vec/2,
            beam_x_positions=x_beam,
            E_d_angle=0,
            drive_hbar_w=probe_hw,
            alpha0_diag=alpha*np.identity(3)[None, ...],
            alpha1_diag=that_alpha*np.identity(3)[None, ...],
            n_b=n_b,
            drive_amp=probe_E,
            return_polarizabilities=False,
            )

        return p1, p2

    # def p_pw_probed(self,
    #     x_beam,
    #     probe_hw,
    #     d,
    #     alpha, that_alpha,
    #     gscale=1,
    #     probe_E=None,
    #     n_b=None):
    #     """ Define dipole magnitude with Gaussian beam driving
    #         force.
    #         """
    #     if probe_E is None:
    #         probe_E = self.E_0

    #     if n_b==None:
    #         n_b=self.n0

    #     d_col_vec = np.array([[d, 0, 0]])

    #     if type(alpha) is np.ndarray:
    #         alpha = np.asarray(alpha).reshape(len(alpha), 1, 1)
    #         that_alpha = np.asarray(that_alpha).reshape(len(alpha), 1, 1)
    #     # else:
    #         # alpha = alpha
    #         # that_alpha = that_alpha


    #     p1, p2 = cp.coupled_dip_mags_both_driven(
    #         mol_angle=0,
    #         plas_angle=0,
    #         d_col=-d_col_vec,
    #         E_d_angle=0,
    #         drive_hbar_w=probe_hw,
    #         alpha0_diag=alpha*np.identity(3)[None, ...],
    #         alpha1_diag=that_alpha*np.identity(3)[None, ...],
    #         n_b=n_b,
    #         drive_amp=probe_E,
    #         return_polarizabilities=False,
    #         )

    #     return p1, p2


    def p1p2_hot_conf(self,
        x_beam,
        probe_hw, T1, b1, T2, b2, d, a1, a2,
        gscale=1,
        **kwargs):
        """ Confocally probed dipole 1"""

        alpha1 = self.alpha_of_T(T1, b1, a1)

        alpha2 = self.alpha_of_T(T2, b2, a2)

        p1, p2 = self.p_gau_probed(
            x_beam,
            # x_dip=-d/2,
            # x_that_dip=+d/2,
            probe_hw=probe_hw,
            d=d,
            alpha=alpha1,
            that_alpha=alpha2,
            gscale=gscale)

        return p1, p2


    def p1p2_cold_conf(self,
        x_beam,
        probe_hw,
        b1, b2, d, a1, a2,
        gscale=1,
        **kwargs):

        return self.p1p2_hot_conf(
            x_beam,
            probe_hw,
            T1=0,
            b1=b1,
            T2=0,
            b2=b2,
            d=d,
            a1=a1,
            a2=a2,
            gscale=gscale,
            **kwargs)

    # def p1p2_hot_wf(self,
    #     x_beam,
    #     probe_hw, T1, b1, T2, b2, d, a1, a2,
    #     gscale=1,
    #     **kwargs):
    #     """ Confocally probed dipole 1"""

    #     alpha1 = self.alpha_of_T(T1, b1, a1)

    #     alpha2 = self.alpha_of_T(T2, b2, a2)

    #     p1, p2 = self.p_pw_probed(
    #         x_beam,
    #         # x_dip=-d/2,
    #         # x_that_dip=+d/2,
    #         probe_hw=probe_hw,
    #         d=d,
    #         alpha=alpha1,
    #         that_alpha=alpha2,
    #         gscale=gscale)

    #     return p1, p2


    # def p1p2_cold_wf(self,
    #     x_beam,
    #     probe_hw,
    #     b1, b2, d, a1, a2,
    #     gscale=1,
    #     **kwargs):

    #     return self.p1p2_hot_conf(
    #         x_beam,
    #         probe_hw,
    #         T1=0,
    #         b1=b1,
    #         T2=0,
    #         b2=b2,
    #         d=d,
    #         a1=a1,
    #         a2=a2,
    #         gscale=gscale,
    #         **kwargs)
    # ## And nor build the scattered power expressions
    # def power_scatt_two_dips(self,
    #     p1, p2,
    #     x_beam,
    #     w, T1, b1, T2, b2, d, a1, a2,
    #     return_components=False,
    #     gscale=1):
    #     """ With interference contributions a la Draine """
    #     sigma_scat_coupled(
    #         dipoles_moments_per_omega,
    #         d_col,
    #         drive_hbar_w,
    #         n_b=None,
    #         E_0=None,
    #         )
    #     t1 = (1/3)*(w**4./c**3.)*np.abs(p1)**2.
    #     t2 = (1/3)*(w**4./c**3.)*np.abs(p2)**2.
    #     t3 = w*np.imag(self.g(0, d, w, gscale))*np.real(p1*np.conj(p2))

    #     if not return_components:
    #         return t1 + t2 + t3
    #     elif return_components:
    #         return np.array([t1, t2, t3])


    def conf_PTI(self,
        x_beam,
        probe_hbar_w, T1, b1, T2, b2, d, a1, a2,
        **kwargs):
        """ Args are all simple floats. Lengths are in units of nm """

        x_beam, d, b1, b2, a1, a2 =  1e-7*np.array([x_beam, d, b1, b2, a1, a2])

        hot_dipoles = lambda omega: self.p1p2_hot_conf(
            x_beam,
            probe_hw=omega*hbar,
            T1=T1,
            b1=b1,
            T2=T2,
            b2=b2,
            d=d,
            a1=a1,
            a2=a2,
            gscale=1,
            **kwargs)

        cold_dipoles = lambda omega: self.p1p2_cold_conf(
            x_beam,
            probe_hw=omega*hbar,
            b1=b1,
            b2=b2,
            d=d,
            a1=a1,
            a2=a2,
            gscale=1,
            **kwargs)


        d_col = np.array([[-d, 0, 0]])
        # hot_signal = self.power_scatt_two_dips(
        #     p1_hot,
        #     p2_hot,
        #     x_beam,
        #     w, T1, b1, T2, b2, d, a1, a2, **kwargs)
        hot_signal, hot_components = cp.sigma_scat_coupled(
            hot_dipoles,
            d_col,
            probe_hbar_w,
            n_b=self.n0,
            E_0=self.E_0,
            )

        cold_signal, cold_components = cp.sigma_scat_coupled(
            cold_dipoles,
            d_col,
            probe_hbar_w,
            n_b=self.n0,
            E_0=self.E_0,
            )

        if 'return_components' in kwargs and kwargs['return_components']:
            return hot_components, cold_components
        else:
            return hot_signal - cold_signal


