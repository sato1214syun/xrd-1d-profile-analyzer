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

        # Estimate Lattice Parameter from Strongest Peak (Cubic only for now)
        if self.lattice.crystal_system.lower() == "cubic":
            self._estimate_cubic_lattice()

    def _estimate_cubic_lattice(self):
        """Estimate cubic lattice parameter 'a' from the strongest observed peak."""
        # Find 2theta of max intensity
        idx_max = np.argmax(self.y)
        two_theta_max = self.x[idx_max]

        # Assume strongest peak is (110) for BCC (Im-3m)
        # Check space group or just assume (110) as it's usually the first strong one
        # For NbTi (Im-3m), (110) is the first allowed reflection.

        # Calculate d-spacing
        # lambda = 2d sin(theta) -> d = lambda / (2 sin(theta))
        theta_rad = np.radians(two_theta_max / 2)
        d_spacing = self.wavelength_val / (2 * np.sin(theta_rad))

        # For (110): d = a / sqrt(1^2 + 1^2 + 0^2) = a / sqrt(2)
        # a = d * sqrt(2)
        a_est = d_spacing * np.sqrt(2)

        print(
            f"Estimated lattice parameter 'a' from max peak at {two_theta_max:.2f} deg: {a_est:.4f} A"
        )

        # Update lattice object
        # For cubic, setting 'a' automatically updates 'b' and 'c' if they are constrained
        self.lattice.a = a_est
        # self.lattice.b = a_est # Error: b is not free
        # self.lattice.c = a_est # Error: c is not free

        # Re-generate reflections because Q might change significantly?
        # Actually, indices don't change for cubic, but Q values do.
        # We should re-generate to be safe, especially if ordering changes (unlikely for cubic).
        self.reflections = self._generate_reflections()

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
        # Increased initial W to ensure peak overlap during initial refinement
        # Reduced max U to prevent excessive broadening at high angles
        params.add("U", value=0.1, min=0.0, max=5.0)
        params.add("V", value=0.0, min=-2.0, max=2.0)
        params.add("W", value=0.5, min=0.001, max=5.0)

        # Peak Shape (Pseudo-Voigt mixing)
        # eta = eta_0 + eta_1 * 2theta
        params.add("eta_0", value=0.5, min=0.0, max=1.0)
        params.add("eta_1", value=0.0, min=-0.01, max=0.01, vary=False)

        # Asymmetry
        params.add("asymmetry", value=0.0, min=-1, max=1)

        # Zero shift & Displacement
        # Increased bounds for zero_shift to handle larger offsets
        params.add("zero_shift", value=0.0, min=-2.0, max=2.0)
        params.add("displacement", value=0.0, min=-2.0, max=2.0, vary=False)

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
        # Estimate initial intensity from observed data at the approximate peak position
        # This prevents "intensity starvation" for strong peaks if they start too small
        max_intensity = self.y.max()

        # Temporary lattice for position estimation
        a_init = lat.a
        b_init = lat.b
        c_init = lat.c

        for i, ref in enumerate(self.reflections):
            hkl = ref["hkl_rep"]
            # Estimate position
            two_theta = self._calculate_2theta(
                hkl, a_init, b_init, c_init, lat.alpha, lat.beta, lat.gamma
            )

            # Find nearest observed intensity
            idx = (np.abs(self.x - two_theta)).argmin()
            estimated_I = self.y[idx]

            # Ensure it's not too small (background level) or too large
            if estimated_I < max_intensity * 0.01:
                estimated_I = max_intensity * 0.01

            # Set initial value. Note: I_i is peak height in our model.
            # We allow intensity to vary (Pawley fit) to ensure stability and convergence.
            params.add(f"I_{i}", value=estimated_I, min=0.0, vary=True)

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

        # Mixing parameter
        eta_0 = params["eta_0"].value
        eta_1 = params["eta_1"].value

        asymmetry = params["asymmetry"].value
        zero = params["zero_shift"].value
        displacement = params["displacement"].value

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
            two_theta_orig = self._calculate_2theta(hkl, a, b, c, alpha, beta, gamma)

            # Apply Zero Shift and Displacement
            # Displacement shift: delta(2theta) = displacement * cos(theta)
            # Note: displacement parameter absorbs the -2s/R factor
            theta_rad_orig = np.radians(two_theta_orig / 2.0)
            disp_shift = displacement * np.cos(theta_rad_orig)

            two_theta = two_theta_orig + zero + disp_shift

            if two_theta < self.x.min() - 1 or two_theta > self.x.max() + 1:
                continue

            # Calculate Width
            theta_rad = np.radians(two_theta / 2)
            tan_theta = np.tan(theta_rad)

            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            if fwhm_sq < 1e-6:
                fwhm_sq = 1e-6
            fwhm = np.sqrt(fwhm_sq)

            # Calculate Eta (Mixing)
            eta = eta_0 + eta_1 * two_theta
            if eta > 1.0:
                eta = 1.0
            if eta < 0.0:
                eta = 0.0

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

        eta_0 = params["eta_0"].value
        eta_1 = params["eta_1"].value

        asymmetry = params["asymmetry"].value
        zero = params["zero_shift"].value
        displacement = params["displacement"].value

        for i, ref in enumerate(self.reflections):
            old_intensity = params[f"I_{i}"].value

            # If intensity is effectively zero, it might stay zero, but Le Bail allows it to grow
            # if there is observed intensity. However, if old_intensity is 0, y_calc_k is 0.
            # So we need to be careful. If we start with non-zero, it's fine.

            hkl = ref["hkl_rep"]

            # Calculate position with displacement
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

            # Calculate Eta
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

            dx_masked = dx[mask]
            sigma_vec = np.where(dx_masked < 0, sigma_l, sigma_r)
            G = np.exp(-np.log(2) * (dx_masked / sigma_vec) ** 2)
            L = 1.0 / (1.0 + (dx_masked / sigma_vec) ** 2)
            peak_shape = eta * L + (1 - eta) * G

            # Le Bail Extraction Formula
            # I_new = Sum_i [ (y_obs(i) - y_bkg(i)) * (y_calc_k(i) / y_calc_peaks_total(i)) ]
            # Here, y_calc_k(i) = old_intensity * peak_shape(i)
            # So, I_new = old_intensity * Sum_i [ peak_shape(i) * ratio(i) ]

            # This I_new is the "Integrated Intensity" (Area) of the peak.
            # However, our model parameter I_i is defined as the "Peak Height" (scaling factor of peak_shape).
            # So we need to convert the extracted Area back to Height.

            sum_term = np.sum(ratio[mask] * peak_shape)
            extracted_area = old_intensity * sum_term

            # Convert Area to Height
            # Area = Height * Integral(peak_shape)
            # Height = Area / Integral(peak_shape)
            shape_integral = np.sum(peak_shape)

            if shape_integral > 1e-9:
                new_intensity = extracted_area / shape_integral
            else:
                new_intensity = 0.0

            # Damping factor to prevent oscillation
            # I_updated = alpha * I_new + (1 - alpha) * I_old
            # Reduced alpha to 0.5 to stabilize intensity extraction
            alpha = 0.5
            new_intensity = alpha * new_intensity + (1 - alpha) * old_intensity

            # Update parameter
            if new_intensity < 0:
                new_intensity = 0
            params[f"I_{i}"].value = new_intensity

    def fit(self, max_nfev=1000, le_bail_cycles=50):
        params = self.make_params()

        # Initial Strategy: Fix Zero Shift and Displacement to ensure Lattice Parameter converges first
        # This prevents the solver from using large zero shifts to fit the first peak while ignoring others.
        params["zero_shift"].set(vary=False, value=0.0)
        params["displacement"].set(vary=False, value=0.0)

        # Weights: 1/sqrt(y) (Poisson statistics)
        # Avoid division by zero
        weights = 1.0 / np.sqrt(np.maximum(self.y, 1.0))

        # Handle Spline Background
        if self.background_type == "spline":
            knot_positions = np.linspace(
                self.x.min(), self.x.max(), self.num_knots + 2
            )[1:-1]
            self.bg_model = SplineModel(prefix="bg_", xknots=knot_positions)
            bg_params = self.bg_model.guess(self.y, x=self.x)
            params.update(bg_params)

        # Pass weights to Minimizer?
        # Minimizer(residual, params, fcn_args=..., fcn_kws=...)
        # But residual() takes only params.
        # We can modify residual to use weights if we pass them, but simpler to just use them in residual.
        # But residual signature is fixed to (params, *args, **kws).
        # Let's store weights in self.
        self.weights = weights

        # Define residual with weights
        def weighted_residual(params):
            res = self.residual(params)
            return res * self.weights

        minner = Minimizer(weighted_residual, params)

        # Le Bail Iteration Loop
        print(f"Starting Le Bail refinement with {le_bail_cycles} cycles...")

        for cycle in range(le_bail_cycles):
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
                # Enable Zero Shift (and Displacement if applicable)
                # Once lattice is roughly correct, we can allow small shifts
                params["zero_shift"].set(vary=True)
                print("  -> Enabling Zero Shift refinement")

                # Enable Asymmetry
                params["asymmetry"].set(vary=True)
                print("  -> Enabling Asymmetry refinement")

            if cycle == 10:
                # Enable Displacement if enough peaks (heuristic)
                # Even with few peaks, if we have high angle data, displacement is better than zero shift alone
                if len(self.reflections) >= 3:
                    params["displacement"].set(vary=True)
                    print("  -> Enabling Displacement refinement")
                else:
                    print("  -> Skipping Displacement (too few reflections)")

            # 1. Refine Geometry, Profile & Intensities (Pawley Fit)
            # Since Intensities are free, we don't need the manual Le Bail extraction step.
            # This avoids the instability of fixed-intensity profile refinement.
            result = minner.minimize(
                method="leastsq", params=params, max_nfev=max_nfev // le_bail_cycles
            )
            params = result.params

            # 2. Extract Intensities (Skipped - using Pawley refinement)
            # self._update_intensities(params)

            print(f"Cycle {cycle + 1}/{le_bail_cycles}: Chi2 = {result.chisqr:.2f}")

        # Final refinement
        result = minner.minimize(method="leastsq", params=params, max_nfev=max_nfev)

        # Calculate R-factors
        y_calc = self.model(result.params)
        if self.background_type == "spline" and hasattr(self, "bg_model"):
            y_calc += self.bg_model.eval(result.params, x=self.x)

        # R_wp (Weighted Profile R-factor)
        # Numerator is essentially Chi2 (unreduced)
        numerator_wp = np.sum((self.weights * (self.y - y_calc)) ** 2)
        denominator_wp = np.sum((self.weights * self.y) ** 2)
        r_wp = np.sqrt(numerator_wp / denominator_wp) * 100

        # R_p (Profile R-factor)
        numerator_p = np.sum(np.abs(self.y - y_calc))
        denominator_p = np.sum(self.y)
        r_p = (numerator_p / denominator_p) * 100

        # R_exp (Expected R-factor)
        # R_exp = sqrt( (N - P) / Sum(w * y_obs^2) )
        # N = number of points, P = number of free parameters
        n_free = result.nfree
        r_exp = np.sqrt(n_free / denominator_wp) * 100

        # Goodness of Fit (S)
        gof = r_wp / r_exp

        # Attach to result object
        result.r_wp = r_wp
        result.r_p = r_p
        result.r_exp = r_exp
        result.gof = gof

        # Calculate Peak Residuals
        result.peak_stats = self.get_peak_stats(result.params)

        return result

    def get_peak_stats(self, params):
        """
        Calculate statistics for each peak: position, intensity, residual.
        """
        a = params["a"].value
        b = params["b"].value
        c = params["c"].value
        alpha = params["alpha"].value
        beta = params["beta"].value
        gamma = params["gamma"].value
        zero = params["zero_shift"].value
        displacement = params["displacement"].value

        # Calculate full profile first
        y_calc = self.model(params)
        if self.background_type == "spline" and hasattr(self, "bg_model"):
            y_calc += self.bg_model.eval(params, x=self.x)

        stats = []

        for i, ref in enumerate(self.reflections):
            hkl = ref["hkl_rep"]

            # Calculate position
            two_theta_orig = self._calculate_2theta(hkl, a, b, c, alpha, beta, gamma)
            theta_rad_orig = np.radians(two_theta_orig / 2.0)
            disp_shift = displacement * np.cos(theta_rad_orig)
            two_theta = two_theta_orig + zero + disp_shift

            # Find nearest data point
            if two_theta < self.x.min() or two_theta > self.x.max():
                continue

            idx = (np.abs(self.x - two_theta)).argmin()
            y_obs_val = self.y[idx]
            y_calc_val = y_calc[idx]
            resid = y_obs_val - y_calc_val
            rel_resid = resid / y_obs_val * 100 if y_obs_val > 1e-3 else 0.0

            stats.append(
                {
                    "hkl": hkl,
                    "2theta": two_theta,
                    "y_obs": y_obs_val,
                    "y_calc": y_calc_val,
                    "residual": resid,
                    "rel_residual_percent": rel_resid,
                }
            )

        return stats

    def plot_result(self, result, save_path="data/Figure_1.png"):
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
        plt.savefig(save_path)
        plt.close()
