import numpy as np
from lmfit import Minimizer
from lmfit.models import SplineModel

try:
    from ..peak_models import split_pseudo_voigt
except ImportError:
    from peak_models import split_pseudo_voigt

from .base import WPPFBase


class LeBailFitter(WPPFBase):
    def _update_intensities(self, params):
        """
        Perform one cycle of Le Bail intensity extraction.
        I_new = Sum_i [ (y_obs(i) - y_bkg(i)) * (y_calc_k(i) / y_calc_peaks_total(i)) ]
        """
        # 1. Calculate Background
        params_bkg = params.copy()
        for i in range(len(self.reflections)):
            params_bkg[f"I_{i}"].value = 0.0

        y_bkg = self.model(params_bkg)
        if self.background_type == "spline" and hasattr(self, "bg_model"):
            y_bkg += self.bg_model.eval(params, x=self.x)

        # 2. Calculate Total Calculated Profile
        y_calc_total = self.model(params)
        if self.background_type == "spline" and hasattr(self, "bg_model"):
            y_calc_total += self.bg_model.eval(params, x=self.x)

        y_peaks_total = y_calc_total - y_bkg

        mask_nonzero = y_peaks_total > 1e-6

        ratio = np.zeros_like(self.x)
        ratio[mask_nonzero] = (
            self.y[mask_nonzero] - y_bkg[mask_nonzero]
        ) / y_peaks_total[mask_nonzero]

        # 3. Iterate over peaks and update intensities
        a = params["a"].value
        b = params["b"].value
        c = params["c"].value
        alpha = params["alpha"].value
        beta = params["beta"].value
        gamma = params["gamma"].value

        U = params["U"].value
        V = params["V"].value
        W = params["W"].value

        eta_0 = params["eta_0"].value
        eta_1 = params["eta_1"].value

        asymmetry = params["asymmetry"].value
        zero = params["zero_shift"].value
        displacement = params["displacement"].value

        for i, ref in enumerate(self.reflections):
            old_intensity = params[f"I_{i}"].value
            hkl = ref["hkl_rep"]

            two_theta_orig = self._calculate_2theta(hkl, a, b, c, alpha, beta, gamma)
            theta_rad_orig = np.radians(two_theta_orig / 2.0)
            disp_shift = displacement * np.cos(theta_rad_orig)
            two_theta = two_theta_orig + zero + disp_shift

            if two_theta < self.x.min() - 1 or two_theta > self.x.max() + 1:
                continue

            theta_rad = np.radians(two_theta / 2)
            tan_theta = np.tan(theta_rad)
            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            if fwhm_sq < 1e-6:
                fwhm_sq = 1e-6
            fwhm = np.sqrt(fwhm_sq)

            eta = eta_0 + eta_1 * two_theta
            if eta > 1.0:
                eta = 1.0
            if eta < 0.0:
                eta = 0.0

            hwhm = fwhm / 2.0
            sigma_l = hwhm * (1 - asymmetry)
            sigma_r = hwhm * (1 + asymmetry)

            dx = self.x - two_theta
            mask = np.abs(dx) < fwhm * 10
            if not np.any(mask):
                continue

            peak_shape = split_pseudo_voigt(
                self.x[mask],
                amplitude=1.0,
                center=two_theta,
                sigma_l=sigma_l,
                sigma_r=sigma_r,
                fraction=eta,
            )

            sum_term = np.sum(ratio[mask] * peak_shape)
            extracted_area = old_intensity * sum_term

            shape_integral = np.sum(peak_shape)

            if shape_integral > 1e-9:
                new_intensity = extracted_area / shape_integral
            else:
                new_intensity = 0.0

            # Clip the update factor to prevent runaway growth/shrinkage
            if old_intensity > 1e-6:
                factor = new_intensity / old_intensity
                factor = np.clip(factor, 0.5, 1.5)
                new_intensity = old_intensity * factor

            # Damping factor to prevent oscillation
            # I_updated = alpha * I_new + (1 - alpha) * I_old
            # Reduced alpha to 0.3 to stabilize intensity extraction
            alpha_damping = 0.3
            new_intensity = (
                alpha_damping * new_intensity + (1 - alpha_damping) * old_intensity
            )

            if new_intensity < 0:
                new_intensity = 0
            params[f"I_{i}"].value = new_intensity

    def fit(self, max_nfev=1000, cycles=50):
        params = self.make_params()

        # For Le Bail, intensities are fixed during profile refinement
        for i in range(len(self.reflections)):
            params[f"I_{i}"].set(vary=False)

        params["zero_shift"].set(vary=False, value=0.0)
        params["displacement"].set(vary=False, value=0.0)

        weights = 1.0 / np.sqrt(np.maximum(self.y, 1.0))

        if self.background_type == "spline":
            knot_positions = np.linspace(
                self.x.min(), self.x.max(), self.num_knots + 2
            )[1:-1]
            self.bg_model = SplineModel(prefix="bg_", xknots=knot_positions)
            bg_params = self.bg_model.guess(self.y, x=self.x)
            # Fix background parameters to prevent instability during Le Bail
            for p in bg_params:
                bg_params[p].set(vary=False)
            params.update(bg_params)

        self.weights = weights

        def weighted_residual(params):
            res = self.residual(params)
            return res * self.weights

        minimizer = Minimizer(weighted_residual, params)

        print(f"Starting Le Bail refinement with {cycles} cycles...")

        # Ensure enough iterations per cycle
        nfev_per_cycle = max(100, max_nfev // cycles)

        for cycle in range(cycles):
            if cycle == 0:
                params["U"].set(vary=False)
                params["V"].set(vary=False)
                params["W"].set(vary=False)
                print("  -> Fixing Width parameters for initial lattice refinement")

            if cycle == 3:
                params["U"].set(vary=True)
                params["V"].set(vary=True)
                params["W"].set(vary=True)
                print("  -> Releasing Width parameters")

            if cycle == 5:
                params["zero_shift"].set(vary=True)
                print("  -> Enabling Zero Shift refinement")
                params["asymmetry"].set(vary=True)
                print("  -> Enabling Asymmetry refinement")

            if cycle == 10:
                if len(self.reflections) >= 3:
                    params["displacement"].set(vary=True)
                    print("  -> Enabling Displacement refinement")
                else:
                    print("  -> Skipping Displacement (too few reflections)")

            # 1. Refine Geometry & Profile (Intensities Fixed)
            result = minimizer.minimize(
                method="leastsq", params=params, max_nfev=nfev_per_cycle
            )
            params = result.params

            # 2. Extract Intensities
            self._update_intensities(params)

            print(f"Cycle {cycle + 1}/{cycles}: Chi2 = {result.chisqr:.2f}")

        # Final refinement
        result = minimizer.minimize(method="leastsq", params=params, max_nfev=max_nfev)

        # Calculate R-factors
        y_calc = self.model(result.params)
        if self.background_type == "spline" and hasattr(self, "bg_model"):
            y_calc += self.bg_model.eval(result.params, x=self.x)

        numerator_wp = np.sum((self.weights * (self.y - y_calc)) ** 2)
        denominator_wp = np.sum((self.weights * self.y) ** 2)
        r_wp = np.sqrt(numerator_wp / denominator_wp) * 100

        numerator_p = np.sum(np.abs(self.y - y_calc))
        denominator_p = np.sum(self.y)
        r_p = (numerator_p / denominator_p) * 100

        n_free = result.nfree
        r_exp = np.sqrt(n_free / denominator_wp) * 100
        gof = r_wp / r_exp

        result.r_wp = r_wp
        result.r_p = r_p
        result.r_exp = r_exp
        result.gof = gof
        result.peak_stats = self.get_peak_stats(result.params)

        return result
