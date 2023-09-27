"""
calculates and plots temperature and pressure profiles
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from plotting import savefig, paper_plot
from dataclasses import dataclass


u.set_enabled_equivalencies(u.mass_energy())


@dataclass
class Profiles:
    halotype: str = "NFW"
    haloargs: dict = None
    T0: float = 1.0e6 * u.K
    r: np.ndarray = np.logspace(-3, 1, 100) * u.Mpc
    gamma: np.float = 1.0
    potential: np.ndarray = None
    density: np.ndarray = None

    def __post_init__(self):
        if self.potential is None:
            self.potential = self.get_potential()
        if self.density is None:
            self.density = self.get_density()

        self.temperature = self.get_temperature()
        self.pressure = self.get_pressure()

    def get_potential(self):
        if halotype == "NFW":
            rho_0 = haloargs["rho_0"]
            rs = haloargs["rs"]
            alpha = 1.0
            beta = 3.0

            potential = (
                -4
                * np.pi
                * G
                * rho_0
                * rs**2
                * (np.log(1 + self.r / rs) / (self.r / rs))
            )
        else:
            raise NotImplementedError()
        return potential

    def get_density(self):
        if halotype == "NFW":
            rho_0 = haloargs["rho_0"]
            rs = haloargs["rs"]
            alpha = 1.0
            beta = 3.0

            density = rho_0 / (
                (self.r / rs) ** alpha * (1 + self.r / rs) ** (beta - alpha)
            )
        else:
            raise NotImplementedError()
        return density

    def get_temperature(self):  # gamma polytropic index
        return (
            1 / k_B * mu * m_p * np.outer((1 - self.gamma) / self.gamma, self.potential)
        )

    def get_pressure(self):
        return (
            k_B / (mu * m_p) * np.multiply(self.temperature(), self.density[:, None].T)
        )

    def get_profile(self):
        pass

    def plot_profile(self):
        paper_plot()
        pass
