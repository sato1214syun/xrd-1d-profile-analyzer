import numpy as np
from lmfit import Model


def split_pseudo_voigt(x, amplitude, center, sigma_l, sigma_r, fraction):
    """
    Split Pseudo-Voigt function (Amplitude is Peak Height)
    sigma_l, sigma_r are HWHM
    """
    dx = x - center
    sigma = np.where(dx < 0, sigma_l, sigma_r)

    # Gaussian part (HWHM = sigma)
    # exp(-ln(2) * (x-c)^2 / sigma^2)
    G = np.exp(-np.log(2) * (dx / sigma) ** 2)

    # Lorentzian part (HWHM = sigma)
    # 1 / (1 + (x-c)^2 / sigma^2)
    L = 1 / (1 + (dx / sigma) ** 2)

    return amplitude * (fraction * L + (1 - fraction) * G)


class SplitPseudoVoigtModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(split_pseudo_voigt, *args, **kwargs)
        self.set_param_hint("fraction", value=0.5, min=0, max=1)
        self.set_param_hint("amplitude", min=0)
        self.set_param_hint("sigma_l", min=0)
        self.set_param_hint("sigma_r", min=0)
        # fwhm expression causes issues with prefixes in some lmfit versions
        # self.set_param_hint("fwhm", expr="sigma_l + sigma_r")
