import numpy as np
import polars as pl
from pathlib import Path
import xrayutilities as xu
from lmfit import Parameters, Minimizer, report_fit
from lmfit.models import PseudoVoigtModel, LinearModel, SplineModel
import matplotlib.pyplot as plt
from peak_fitting import SplitPseudoVoigtModel


class LeBailFitter:
    def __init__(
        self,
        x,
        y,
        cif_path,
        wavelength="CuKa12",
        background_type="spline",
        num_knots=10,
    ):
        """
        Le Bail / Pawley fitting class.

        Parameters
        ----------
        x : array_like
            2theta values
        y : array_like
            Intensity values
        cif_path : str or Path
            Path to CIF file
        wavelength : str
            X-ray wavelength source (e.g. "CuKa1")
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.cif_path = Path(cif_path)
        self.wavelength_name = wavelength
        self.wavelength_val = xu.wavelength(wavelength)
        self.background_type = background_type
        self.num_knots = num_knots

        # Load Crystal
        if not self.cif_path.exists():
            raise FileNotFoundError(f"CIF not found: {self.cif_path}")
        self.crystal = xu.materials.Crystal.fromCIF(str(self.cif_path))
        self.lattice = self.crystal.lattice

        print(f"Loaded Crystal: {self.crystal.name}")
        print(f"System: {self.lattice.crystal_system}")
        print(f"Space Group: {self.lattice.space_group}")

        # Generate Reflections
        self.reflections = self._generate_reflections()
        print(f"Generated {len(self.reflections)} unique reflections.")

    def _generate_reflections(self):
        """Generate unique reflections (HKL groups) for the measured range."""
        q_max = 4 * np.pi / self.wavelength_val * np.sin(np.radians(self.x.max() / 2))

        # Get all allowed HKLs
        hkls_set = self.lattice.get_allowed_hkl(q_max)
        hkl_list = list(hkls_set)

        # Calculate Q for each to group them
        # We use the initial lattice parameters for grouping
        # Grouping by Q implies grouping by d-spacing

        grouped_reflections = {}

        for hkl in hkl_list:
            q = self.lattice.GetQ(hkl)
            q_mag = np.linalg.norm(q)

            # Round Q to group equivalents (tolerance for floating point)
            q_key = round(q_mag, 5)

            if q_key not in grouped_reflections:
                grouped_reflections[q_key] = {
                    "hkl_list": [],
                    "hkl_rep": hkl,  # Representative
                    "q_mag": q_mag,
                    "multiplicity": 0,
                }

            grouped_reflections[q_key]["hkl_list"].append(hkl)
            grouped_reflections[q_key]["multiplicity"] += 1

        # Sort by Q (2theta)
        sorted_refs = sorted(grouped_reflections.values(), key=lambda r: r["q_mag"])

        # Filter out peaks below min 2theta
        q_min = 4 * np.pi / self.wavelength_val * np.sin(np.radians(self.x.min() / 2))
        sorted_refs = [r for r in sorted_refs if r["q_mag"] >= q_min]

        return sorted_refs

    def _calculate_2theta(self, hkl, a, b, c, alpha, beta, gamma):
        """
        Calculate 2theta for a given HKL and lattice parameters.
        This manually calculates d-spacing to avoid updating the global crystal object repeatedly.
        """
        # Convert angles to radians
        ar = np.radians(alpha)
        br = np.radians(beta)
        gr = np.radians(gamma)

        # Metric Tensor (Real Space)
        cos_a = np.cos(ar)
        cos_b = np.cos(br)
        cos_g = np.cos(gr)

        G = np.array(
            [
                [a**2, a * b * cos_g, a * c * cos_b],
                [a * b * cos_g, b**2, b * c * cos_a],
                [a * c * cos_b, b * c * cos_a, c**2],
            ]
        )

        try:
            G_star = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return 0.0  # Invalid lattice

        h, k, l = hkl
        hkl_vec = np.array([h, k, l])

        # 1/d^2 = hkl . G* . hkl
        inv_d2 = hkl_vec @ G_star @ hkl_vec

        if inv_d2 <= 0:
            return 0.0

        d = 1.0 / np.sqrt(inv_d2)

        # Bragg law: lambda = 2d sin(theta)
        # sin(theta) = lambda / 2d
        sin_theta = self.wavelength_val / (2 * d)

        # Check domain
        if sin_theta > 1.0:
            return 180.0  # Unphysical

        theta = np.arcsin(sin_theta)
        return np.degrees(2 * theta)

    def make_params(self):
        params = Parameters()

        # Lattice Parameters
        lat = self.lattice
        sys = lat.crystal_system.lower()

        # Base parameters
        params.add("a", value=lat.a, min=lat.a * 0.9, max=lat.a * 1.1)
        params.add("b", value=lat.b, min=lat.b * 0.9, max=lat.b * 1.1)
        params.add("c", value=lat.c, min=lat.c * 0.9, max=lat.c * 1.1)
        params.add("alpha", value=lat.alpha, min=80, max=100, vary=False)
        params.add("beta", value=lat.beta, min=80, max=100, vary=False)
        params.add("gamma", value=lat.gamma, min=80, max=100, vary=False)

        # Constraints based on system
        if sys == "cubic":
            params["b"].set(expr="a")
            params["c"].set(expr="a")
        elif sys == "tetragonal":
            params["b"].set(expr="a")
        elif sys == "hexagonal" or sys == "trigonal":
            params["b"].set(expr="a")
            params["gamma"].set(value=120, vary=False)
        elif sys == "orthorhombic":
            pass  # all free
        elif sys == "monoclinic":
            # Assuming standard setting where beta != 90
            if abs(lat.beta - 90) > 0.1:
                params["beta"].set(vary=True)
            elif abs(lat.alpha - 90) > 0.1:
                params["alpha"].set(vary=True)
            elif abs(lat.gamma - 90) > 0.1:
                params["gamma"].set(vary=True)
        elif sys == "triclinic":
            params["alpha"].set(vary=True)
            params["beta"].set(vary=True)
            params["gamma"].set(vary=True)

        # Profile Parameters (Caglioti)
        # FWHM^2 = U tan^2(th) + V tan(th) + W
        # Relaxed constraints and larger initial values
        params.add("U", value=0.1, min=0.0, max=2.0)
        params.add("V", value=0.0, min=-1.0, max=1.0)
        params.add("W", value=0.1, min=0.001, max=1.0)

        # Peak Shape (Pseudo-Voigt mixing)
        params.add("eta", value=0.5, min=0.0, max=1.0)

        # Asymmetry
        params.add("asymmetry", value=0.0, min=-1, max=1)

        # Zero shift
        params.add("zero_shift", value=0.0, min=-0.5, max=0.5)

        # Background
        if self.background_type == "linear":
            params.add("bg_slope", value=0.0)
            params.add("bg_intercept", value=self.y.min())
        elif self.background_type == "polynomial":
            # Polynomial background (degree 4)
            params.add("bg_c0", value=self.y.min())
            params.add("bg_c1", value=0.0)
            params.add("bg_c2", value=0.0)
            params.add("bg_c3", value=0.0)
            params.add("bg_c4", value=0.0)
        elif self.background_type == "spline":
            # Spline background is handled by adding a SplineModel to the composite model
            # We don't add parameters here manually for spline, but we need to handle it in fit()
            pass

        # Intensities (Le Bail parameters)
        # One per reflection group
        # In Le Bail method, intensities are not refined by least-squares but extracted iteratively.
        # So we set vary=False initially.
        for i, ref in enumerate(self.reflections):
            params.add(f"I_{i}", value=100.0, min=0.0, vary=False)

        return params

    def model(self, params):
        """Calculate the full pattern."""
        # Extract lattice params
        a = params["a"].value
        b = params["b"].value
        c = params["c"].value
        alpha = params["alpha"].value
        beta = params["beta"].value
        gamma = params["gamma"].value

        U = params["U"].value
        V = params["V"].value
        W = params["W"].value
        eta = params["eta"].value
        asymmetry = params["asymmetry"].value
        zero = params["zero_shift"].value

        y_calc = np.zeros_like(self.x)

        # Background
        if self.background_type == "linear":
            slope = params["bg_slope"].value
            intercept = params["bg_intercept"].value
            y_calc += slope * self.x + intercept
        elif self.background_type == "polynomial":
            c0 = params["bg_c0"].value
            c1 = params["bg_c1"].value
            c2 = params["bg_c2"].value
            c3 = params["bg_c3"].value
            c4 = params["bg_c4"].value
            # Normalize x to [-1, 1] for stability
            x_norm = 2 * (self.x - self.x.min()) / (self.x.max() - self.x.min()) - 1
            y_calc += (
                c0 + c1 * x_norm + c2 * x_norm**2 + c3 * x_norm**3 + c4 * x_norm**4
            )
        elif self.background_type == "spline":
            # For spline, we assume the parameters are already in 'params' with prefix 'bg_'
            # We need to reconstruct the SplineModel to evaluate it
            # This is inefficient to do every iteration if we recreate the model
            # Instead, we should use the lmfit Model interface properly
            pass

        # Peaks
        for i, ref in enumerate(self.reflections):
            intensity = params[f"I_{i}"].value
            if intensity <= 1e-5:
                continue

            hkl = ref["hkl_rep"]

            # Calculate position
            two_theta = self._calculate_2theta(hkl, a, b, c, alpha, beta, gamma)
            two_theta += zero

            if two_theta < self.x.min() - 1 or two_theta > self.x.max() + 1:
                continue

            # Calculate Width
            theta_rad = np.radians(two_theta / 2)
            tan_theta = np.tan(theta_rad)

            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            if fwhm_sq < 1e-6:
                fwhm_sq = 1e-6
            fwhm = np.sqrt(fwhm_sq)

            # Split Pseudo-Voigt parameters
            # sigma_l and sigma_r are HWHM
            hwhm = fwhm / 2.0
            sigma_l = hwhm * (1 - asymmetry)
            sigma_r = hwhm * (1 + asymmetry)

            # Use the split_pseudo_voigt function from peak_fitting module logic
            # Re-implementing here for speed inside the loop or using the Model class?
            # Using the function directly is faster than creating Model objects

            dx = self.x - two_theta
            # Optimization: only calculate near peak
            mask = np.abs(dx) < fwhm * 10
            if not np.any(mask):
                continue

            dx_masked = dx[mask]

            # Split Pseudo-Voigt Calculation
            # sigma depends on side
            sigma_vec = np.where(dx_masked < 0, sigma_l, sigma_r)

            # Gaussian part
            G = np.exp(-np.log(2) * (dx_masked / sigma_vec) ** 2)

            # Lorentzian part
            L = 1.0 / (1.0 + (dx_masked / sigma_vec) ** 2)

            peak_shape = eta * L + (1 - eta) * G

            y_calc[mask] += intensity * peak_shape

        return y_calc

    def residual(self, params):
        # If using spline, we need to evaluate it here
        res = self.y - self.model(params)

        if self.background_type == "spline":
            # Evaluate spline background
            # We need to find the spline parameters in 'params'
            # and evaluate the spline.
            # Since SplineModel in lmfit is a bit complex to evaluate manually without the object,
            # we might need to keep the SplineModel instance.
            if hasattr(self, "bg_model"):
                bg_val = self.bg_model.eval(params, x=self.x)
                res -= bg_val

        return res

    def _update_intensities(self, params):
        """
        Perform one cycle of Le Bail intensity extraction.
        I_new = Sum_i [ (y_obs(i) - y_bkg(i)) * (y_calc_k(i) / y_calc_peaks_total(i)) ]
        """
        # 1. Calculate Background
        # Create params with 0 intensity to get background only
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

        # Avoid division by zero
        # If y_peaks_total is very small, the ratio is unstable.
        # We can set a threshold.
        mask_nonzero = y_peaks_total > 1e-6

        ratio = np.zeros_like(self.x)
        ratio[mask_nonzero] = (
            self.y[mask_nonzero] - y_bkg[mask_nonzero]
        ) / y_peaks_total[mask_nonzero]

        # 3. Iterate over peaks and update intensities
        # We need to recalculate peak shapes.
        # This duplicates logic from model(), but we need individual components.

        a = params["a"].value
        b = params["b"].value
        c = params["c"].value
        alpha = params["alpha"].value
        beta = params["beta"].value
        gamma = params["gamma"].value

        U = params["U"].value
        V = params["V"].value
        W = params["W"].value
        eta = params["eta"].value
        asymmetry = params["asymmetry"].value
        zero = params["zero_shift"].value

        for i, ref in enumerate(self.reflections):
            old_intensity = params[f"I_{i}"].value

            # If intensity is effectively zero, it might stay zero, but Le Bail allows it to grow
            # if there is observed intensity. However, if old_intensity is 0, y_calc_k is 0.
            # So we need to be careful. If we start with non-zero, it's fine.

            hkl = ref["hkl_rep"]
            two_theta = self._calculate_2theta(hkl, a, b, c, alpha, beta, gamma)
            two_theta += zero

            if two_theta < self.x.min() - 1 or two_theta > self.x.max() + 1:
                continue

            theta_rad = np.radians(two_theta / 2)
            tan_theta = np.tan(theta_rad)
            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            if fwhm_sq < 1e-6:
                fwhm_sq = 1e-6
            fwhm = np.sqrt(fwhm_sq)

            hwhm = fwhm / 2.0
            sigma_l = hwhm * (1 - asymmetry)
            sigma_r = hwhm * (1 + asymmetry)

            dx = self.x - two_theta
            mask = np.abs(dx) < fwhm * 10
            if not np.any(mask):
                continue

            dx_masked = dx[mask]
            sigma_vec = np.where(dx_masked < 0, sigma_l, sigma_r)
            G = np.exp(-np.log(2) * (dx_masked / sigma_vec) ** 2)
            L = 1.0 / (1.0 + (dx_masked / sigma_vec) ** 2)
            peak_shape = eta * L + (1 - eta) * G

            # sum_term is the extracted "Area" (sum of counts) for this peak
            sum_term = np.sum(ratio[mask] * peak_shape)
            extracted_area = old_intensity * sum_term

            # Convert Area to Height (which is the parameter I_i)
            # Height = Area / Sum(Shape)
            shape_integral = np.sum(peak_shape)

            if shape_integral > 1e-9:
                new_intensity = extracted_area / shape_integral
            else:
                new_intensity = 0.0

            # Update parameter
            if new_intensity < 0:
                new_intensity = 0
            params[f"I_{i}"].value = new_intensity

    def fit(self, max_nfev=1000, le_bail_cycles=10):
        params = self.make_params()

        # Handle Spline Background
        if self.background_type == "spline":
            knot_positions = np.linspace(
                self.x.min(), self.x.max(), self.num_knots + 2
            )[1:-1]
            self.bg_model = SplineModel(prefix="bg_", xknots=knot_positions)
            bg_params = self.bg_model.guess(self.y, x=self.x)
            params.update(bg_params)

        minner = Minimizer(self.residual, params)

        # Le Bail Iteration Loop
        print(f"Starting Le Bail refinement with {le_bail_cycles} cycles...")

        for cycle in range(le_bail_cycles):
            # 1. Refine Geometry & Profile (Intensities Fixed)
            # We use fewer steps per cycle to speed up
            # IMPORTANT: Pass the current params to minimize so it uses updated intensities and geometry
            result = minner.minimize(
                method="leastsq", params=params, max_nfev=max_nfev // le_bail_cycles
            )
            params = result.params

            # 2. Extract Intensities
            self._update_intensities(params)

            print(f"Cycle {cycle + 1}/{le_bail_cycles}: Chi2 = {result.chisqr:.2f}")

        # Final refinement
        result = minner.minimize(method="leastsq", params=params, max_nfev=max_nfev)
        return result

    def plot_result(self, result):
        y_calc = self.model(result.params)

        if self.background_type == "spline" and hasattr(self, "bg_model"):
            y_calc += self.bg_model.eval(result.params, x=self.x)

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.y, "k.", label="Observed")
        plt.plot(self.x, y_calc, "r-", label="Calculated")
        plt.plot(self.x, self.y - y_calc, "g-", label="Residual")
        plt.legend()
        plt.xlabel("2Theta")
        plt.ylabel("Intensity")
        plt.title("Le Bail / Pawley Fit Result")
        plt.show()
