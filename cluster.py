import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy.special import gamma


def c(n):
    return 2 ** (5 + n / 2) / (3 * np.sqrt(np.pi)) * gamma(3 + n / 2)


def temp_from_vdisp(vel_disp):
    return (vel_disp ** 2 * const.m_p / const.k_B).to(u.GeV, equivalencies=u.temperature_energy())


class Cluster:
    with u.set_enabled_equivalencies(u.mass_energy()):
        m_b = const.m_p.to(u.GeV)  # baryon particle mass
        m_chi = np.logspace(-5, 3, num=100) * u.GeV
        adiabatic_idx = 5 / 3

    def __init__(
            self, radius, mass, vel_disp, t_c=1 * u.Gyr, accretion=1e-7 * const.M_sun / u.s,
            epsilon=0.01, fb=0.1, fdm=0.9, m500=None, v500=None):
        with u.set_enabled_equivalencies(u.mass_energy()):
            # General - from data
            self.radius = radius  # radius
            self.mass = mass.to(u.GeV)  # total mass
            self.vel_disp = vel_disp  # velocity dispersion

            # General - calculated
            self.volume = 4 / 3 * np.pi * self.radius ** 3  # cluster volume
            self.rho_tot = (self.mass / self.volume).to(u.GeV / u.cm ** 3)  # total density
            self.rho_b = self.rho_tot * fb  # baryon density
            self.rho_dm = self.rho_tot * fdm  # DM density
            self.baryon_temp = temp_from_vdisp(self.vel_disp)  # baryon temperature

            # AGN heating params
            self.m500 = m500
            self.accretion = accretion  # accretion rate
            self.epsilon = epsilon  # efficiency

            # radiative cooling params
            self.n_e = self.rho_b / self.m_b  # number density of electrons
            self.t_c = t_c.to(u.s)  # cooling time

            # to calculate luminosity
            self.v500 = v500
            self.T500 = temp_from_vdisp(self.v500) if v500 is not None else None

    def agn_heating_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            return (self.epsilon * self.accretion_rate() * const.c ** 2).to(u.GeV / u.s,
                                                                             equivalencies=u.temperature_energy())

    def accretion_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            # G = const.G.to(u.cm**3 * u.GeV**(-1) * u.s**(-2), equivalencies=u.mass_energy())
            norm = 1 / 4  # normalization factor of order 1, norm(adiabatic_idx=5/3)=1/4
            mu = 1  # mean molecular weight of gas, 1 for proton gas (hydrogen)
            leading_factors = norm * 4*np.pi *const.c ** -5
            gm2 = (const.G * self.bh_mass()) ** 2
            frac = (mu * self.m_b) ** (5 / 2) / self.adiabatic_idx ** (3 / 2)
            return leading_factors * gm2 * frac * self.plasma_entropy() ** (-3 / 2)

    def bh_mass(self):  # from Gaspari 2019 table 8
        slope = 1.39
        intercept = -9.56 * u.Msun
        return (slope * self.m500 + intercept).to(u.kg)

    def plasma_entropy(self):
        n = (2 * self.n_e).to(u.m ** (-3))  # baryon number density
        return (const.k_B * self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())).to(u.GeV) / n ** (
                self.adiabatic_idx - 1)

    def cooling_time(self):
        t_c = (3 * const.k_B * self.n_e * self.volume * self.baryon_temp.to(
            u.K, equivalencies=u.temperature_energy())) / (2 * (self.luminosity()).to(u.J / u.s))
        return t_c.to(u.s)

    def radiative_cooling_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            return (3 / 2 * self.n_e * self.baryon_temp / self.t_c * self.volume).to(u.GeV / u.s)

    def luminosity(self):
        T = self.T500.to(u.K, equivalencies=u.temperature_energy())
        b = -2.34 * 1e44 * u.erg / u.s
        m = (4.71 * 1e44 * u.erg / u.s) / u.K
        L = m * T + b
        return L.to(u.GeV / u.s)

    def virial_temperature(self, f_chi=1, m_psi=0.1 * u.GeV):
        frac = (f_chi / self.m_chi + (1 - f_chi) / m_psi)
        M_kg = self.mass.to(u.kg, equivalencies=u.mass_energy())
        return (0.3 * const.G * M_kg / (self.radius * frac) * 1 / const.c ** 2).to(u.GeV)

    def sigma0(self, f_chi=1, m_psi=0.1 * u.GeV, n=0):
        with u.set_enabled_equivalencies(u.mass_energy()):
            dm_temp = self.virial_temperature(f_chi=f_chi, m_psi=m_psi)
            uth = np.sqrt(self.baryon_temp / self.m_b + dm_temp / self.m_chi)
            rho_chi = self.rho_dm * f_chi
            total_heating_rate = self.agn_heating_rate() - self.radiative_cooling_rate()
            numerator = total_heating_rate * (self.m_chi + self.m_b) ** 2
            denominator = 3 * (self.baryon_temp - dm_temp) * rho_chi * self.rho_b * self.volume * c(n) * uth ** (
                    (n + 1) / 2) * const.c.to(u.cm / u.s)
            sigma0 = (numerator / denominator).to(u.cm ** 2)
            return sigma0

    def plot_T_chi_vs_m_chi(self, f_chi=1, m_psi=0.1 * u.GeV):
        plt.loglog(self.m_chi, self.virial_temperature(f_chi=f_chi, m_psi=m_psi),
                   label=f'DM temp = virial temp, fx={f_chi}')
        plt.xlabel(r'$m_{\chi} (GeV)$')
        plt.ylabel(r'$T_{\chi} (GeV)$')
        plt.legend(loc='upper left')

    def plot_sigma0_vs_m_chi(self, f_chi=[1], m_psi=[0.1 * u.GeV], n=[0]):
        params = [(f, m, i) for f in f_chi for m in m_psi for i in n]
        for (f, m, i) in params:
            plt.loglog(self.m_chi, self.sigma0(f_chi=f, m_psi=m, n=i),
                       label=f'fx={f}, n={i}, m_psi={m}')
        plt.xlabel(r'$m_{\chi} (GeV)$')
        plt.ylabel(r'$\sigma_0 (cm^2)$')
        plt.legend(loc='upper left')
