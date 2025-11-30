import numpy as np
from lmfit import Minimizer
from lmfit.models import SplineModel

from .base import WPPFBase


class PawleyFitter(WPPFBase):
    def fit(self, max_nfev=1000, cycles=50):
        params = self.make_params()

        # Initial Strategy: Fix Zero Shift and Displacement to ensure Lattice Parameter converges first
        params["zero_shift"].set(vary=False, value=0.0)
        params["displacement"].set(vary=False, value=0.0)
        params["transparency"].set(vary=False, value=0.0)
        params["axial_divergence"].set(vary=False, value=0.0)
        params["error_quad"].set(vary=False, value=0.0)

        # Weights: 1/sqrt(y) (Poisson statistics)
        weights = 1.0 / np.sqrt(np.maximum(self.y, 1.0))

        # Handle Spline Background
        if self.background_type == "spline":
            knot_positions = np.linspace(
                self.x.min(), self.x.max(), self.num_knots + 2
            )[1:-1]
            self.bg_model = SplineModel(prefix="bg_", xknots=knot_positions)
            bg_params = self.bg_model.guess(self.y, x=self.x)
            params.update(bg_params)

        self.weights = weights

        # Define residual with weights
        def weighted_residual(params):
            res = self.residual(params)
            return res * self.weights

        minimizer = Minimizer(weighted_residual, params)

        print(f"Starting Pawley refinement with {cycles} cycles...")

        for cycle in range(cycles):
            # Gradual Parameter Release Strategy

            # Cycle 0-2: Fix Width to prevent collapse, Refine Lattice only
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
                # Increased threshold to 6 to prevent correlation with Zero Shift when peaks are few
                if len(self.reflections) >= 6:
                    params["displacement"].set(vary=True)
                    print("  -> Enabling Displacement refinement")
                else:
                    print("  -> Skipping Displacement (too few reflections)")

            # if cycle == 15:
            #     params["transparency"].set(vary=True)
            #     print("  -> Enabling Transparency refinement")
            #     params["axial_divergence"].set(vary=True)
            #     print("  -> Enabling Axial Divergence refinement")

            # if cycle == 20:
            #     params["error_quad"].set(vary=True)
            #     print("  -> Enabling Quadratic Error refinement")

            # Pawley Fit: Intensities are free parameters
            result = minimizer.minimize(
                method="leastsq", params=params, max_nfev=max_nfev // cycles
            )
            params = result.params

            print(f"Cycle {cycle + 1}/{cycles}: Chi2 = {result.chisqr:.2f}")

        # Final refinement
        # Calculate covariance matrix to get standard errors
        result = minimizer.minimize(method="leastsq", params=params, max_nfev=max_nfev)

        # Check for high correlations
        print("\n--- Parameter Correlations (> 0.8) ---")
        for i, p1 in enumerate(result.params):
            for p2 in list(result.params)[i + 1 :]:
                if result.params[p1].correl and p2 in result.params[p1].correl:
                    corr = result.params[p1].correl[p2]
                    if abs(corr) > 0.8:
                        print(f"{p1} vs {p2}: {corr:.4f}")
        print("--------------------------------------\n")

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
        result.best_fit = y_calc

        return result
