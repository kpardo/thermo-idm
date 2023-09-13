import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy.special import gamma
from scipy.optimize import fsolve


def c(n):
    return 2 ** (5 + n / 2) / (3 * np.sqrt(np.pi)) * gamma(3 + n / 2)


def temp_from_vdisp(vel_disp):
    return (vel_disp ** 2 * const.m_p / const.k_B).to(u.GeV, equivalencies=u.temperature_energy())
    
def func(T_b, p0, cluster):
    #function used to solve for T_b
    sigma0 = p0[0]*u.cm**2
    m_chi=p0[1]*u.GeV
    T_b = T_b*u.GeV
    
    V=cluster.volume.to(u.cm**3)
    x = (3*const.c*c(n)*V*cluster.rho_dm*cluster.rho_b*sigma0/(cluster.m_b+m_chi)**2).to(1/u.s)
    gm2 = ((const.G * cluster.bh_mass()) ** 2).to(u.cm**6/u.s**4)
    frac = ((cluster.mu * cluster.m_b) ** (5 / 2) / cluster.adiabatic_idx ** (3 / 2)).to(u.GeV**(5/2))
    nb = (2 * cluster.n_e).to(u.cm ** (-3)) # baryon number density
    D = (cluster.epsilon*cluster.leading_factors*gm2*frac*(1/nb**(2/3))**(-3/2)) # removed k_B from original function because we are working in GeV here
    T_chi = cluster.virial_temperature(m_chi)
    
    numerator = D*T_b**(-3/2)
    denominator = (T_b - T_chi)*(T_chi/m_chi + T_b/cluster.m_b)**(1/2)
    
    return ((numerator/denominator - x)*const.hbar).to(u.GeV, equivalencies=u.temperature_energy())
    
    

class Cluster:
    with u.set_enabled_equivalencies(u.mass_energy()):
        m_b = const.m_p.to(u.GeV)  # baryon particle mass
        m_chi = np.logspace(-5, 3, num=100) * u.GeV
        adiabatic_idx = 5 / 3
        norm = 1 / 4 # accretion rate normalization factor of order 1, norm(adiabatic_idx=5/3)=1/4
        mu = 1  # mean molecular weight of gas, 1 for proton gas (hydrogen)

    def __init__(
            self, radius, mass, vel_disp, 
            epsilon=0.01, fb=0.1, fdm=0.9, m500=None, v500=0*u.km/u.s):
        with u.set_enabled_equivalencies(u.mass_energy()):
            # General - read from data
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
            self.epsilon = epsilon  # efficiency

            # radiative cooling params
            self.n_e = self.rho_b / self.m_b  # number density of electrons

            # to calculate luminosity
            self.v500 = v500 # default 0 gives no cooling

    def agn_heating_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            return (self.epsilon * self.accretion_rate()).to(u.GeV / u.s,
                                                                             equivalencies=u.temperature_energy())

    def accretion_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            leading_factors = self.norm * 4*np.pi *const.c ** -3
            gm2 = (const.G * self.bh_mass()) ** 2
            frac = (self.mu * self.m_b) ** (5 / 2) / self.adiabatic_idx ** (3 / 2)
            return leading_factors * gm2 * frac * self.plasma_entropy() ** (-3 / 2)

    def bh_mass(self):  # from Gaspari 2019 figure 8
        slope = 1.39
        intercept = -9.56 * u.Msun
        return (slope * self.m500 + intercept).to(u.kg)

    def plasma_entropy(self):
        n = (2 * self.n_e).to(u.m ** (-3))  # baryon number density
        return (const.k_B * self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())).to(u.GeV) / n ** (
                self.adiabatic_idx - 1)

    def cooling_time(self):
        # THIS IS NOT A VALID WAY OF CALCULATING COOLING TIME
        t_c = (3 * const.k_B * self.n_e * self.volume * self.baryon_temp.to(
            u.K, equivalencies=u.temperature_energy())) / (2 * (self.luminosity()).to(u.J / u.s))
        return t_c.to(u.s)

    def radiative_cooling_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            #return (3 / 2 * self.n_e * self.baryon_temp / self.cooling_time() * self.volume).to(u.GeV / u.s)
            return 0 # NO RADIATIVE COOlING RIGHT NOW UNTIL I FIND A BETTER WAY TO CALCULATE

    def luminosity(self): # from Gaspari 2019 figure A1 
        T = temp_from_vdisp(self.v500).to(u.K, equivalencies=u.temperature_energy())
        b = -2.34 * 1e44 * u.erg / u.s
        m = (4.71 * 1e44 * u.erg / u.s) / u.K
        L = m * T + b
        return L.to(u.GeV / u.s)

    def virial_temperature(self, m_chi, f_chi=1, m_psi=0.1 * u.GeV):
        frac = (f_chi / m_chi + (1 - f_chi) / m_psi)
        M_kg = self.mass.to(u.kg, equivalencies=u.mass_energy())
        return (0.3 * const.G * M_kg / (self.radius * frac) * 1 / const.c ** 2).to(u.GeV)

    def sigma0(self, f_chi=1, m_psi=0.1 * u.GeV, n=0):
        with u.set_enabled_equivalencies(u.mass_energy()):
            dm_temp = self.virial_temperature(self.m_chi, f_chi=f_chi, m_psi=m_psi)
            uth = np.sqrt(self.baryon_temp / self.m_b + dm_temp / self.m_chi)
            rho_chi = self.rho_dm * f_chi
            total_heating_rate = self.agn_heating_rate() - self.radiative_cooling_rate()
            numerator = total_heating_rate * (self.m_chi + self.m_b) ** 2
            denominator = 3 * (self.baryon_temp - dm_temp) * rho_chi * self.rho_b * self.volume * c(n) * uth ** (n + 1) * const.c.to(u.cm / u.s)
            sigma0 = (numerator / denominator).to(u.cm ** 2)
            return sigma0

    def plot_T_chi_vs_m_chi(self, f_chi=1, m_psi=0.1 * u.GeV): # produces T_chi vs m_chi plot given an f_chi and m_psi
        plt.loglog(self.m_chi, self.virial_temperature(self.m_chi, f_chi=f_chi, m_psi=m_psi),
                   label=f'DM temp = virial temp, fx={f_chi}')
        plt.xlabel(r'$m_{\chi} (GeV)$')
        plt.ylabel(r'$T_{\chi} (GeV)$')
        plt.legend(loc='upper left')

    def plot_sigma0_vs_m_chi(self, f_chi=[1], m_psi=[0.1 * u.GeV], n=[0], region=False): 
        # plots sigma0 vs m_chi for all combinations of f_chi, m_psi, and n
        params = [(f, m, i) for f in f_chi for m in m_psi for i in n]
        for (f, m, i) in params:
            sigma0 = self.sigma0(f_chi=f, m_psi=m, n=i)
            label = f'fx={f}, n={i}' 
            label = label + f', m_psi={m}' if f<1 else label
            plt.loglog(self.m_chi, sigma0,
                       label=label)
            if region:
                plt.fill_between(self.m_chi.value, sigma0.value, y2=1e-15, alpha=0.3)


        plt.xlabel(r'$m_{\chi} (GeV)$')
        plt.ylabel(r'$\sigma_0 (cm^2)$')
        plt.legend(loc='upper left')

#model testing methods:
    def pred_T_b_small_m(self, sigma0, m_chi):
        # approximates T_b for small m_chi -> T_chi~0
        V=self.volume.to(u.cm**3)
        x = (3*const.c*c(n)*V*self.rho_dm*self.rho_b*sigma0/(self.m_b+m_chi)**2).to(1/u.s)
        leading_factors = (self * 4*np.pi *const.c ** -3).to(u.s**3/u.cm**3)
        gm2 = ((const.G * self.bh_mass()) ** 2).to(u.cm**6/u.s**4)
        frac = ((self.mu * self.m_b) ** (5 / 2) / self.adiabatic_idx ** (3 / 2)).to(u.GeV**(5/2))
        nb = (2 * self.n_e).to(u.cm ** (-3)) # baryon number density
        D = (self.epsilon*leading_factors*gm2*frac*(1/nb**(2/3))**(-3/2)) # removed k_B from original function because we are working in GeV here
        T_b = (((D*np.sqrt(self.m_b))/x)**(1/3)).to(u.GeV, equivalencies=u.temperature_energy())
        return T_b

    def pred_T_b(self, p0): #p0 is a vector with p0[0] = log(sigma0) and p0[1]=log(m_chi)
        x0 = 1e-6 * u.GeV # starting estimate (could even do this using T_b_small)
        return fsolve(func, x0, args=(p0, self))*u.GeV
