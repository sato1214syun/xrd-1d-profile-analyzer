from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xrayutilities as xu
from lmfit import Parameters

try:
    from ..peak_models import split_pseudo_voigt
except ImportError:
    from peak_models import split_pseudo_voigt


class WPPFBase:
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
        Base class for Whole Powder Pattern Fitting (WPPF).

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

        # Estimate Lattice Parameter from Strongest Peak (Generic)
        self._estimate_lattice_from_strongest_peak()

    def _estimate_lattice_from_strongest_peak(self):
        """
        Estimate lattice parameters by matching the strongest observed peak
        to the strongest theoretical reflection.
        Scales the unit cell isotropically.
        """
        # 1. Find strongest observed peak
        idx_max = np.argmax(self.y)
        two_theta_max = self.x[idx_max]
        theta_rad_max = np.radians(two_theta_max / 2)
        d_obs = self.wavelength_val / (2 * np.sin(theta_rad_max))

        print(f"Strongest observed peak at {two_theta_max:.2f} deg (d={d_obs:.4f} A)")

        # 2. Calculate theoretical intensities to find strongest reflection
        # Use a simplified approach: F^2 * LP * Multiplicity

        # Calculate max Q for the whole range
        q_min = 4 * np.pi / self.wavelength_val * np.sin(np.radians(self.x.min() / 2))
        q_max = 4 * np.pi / self.wavelength_val * np.sin(np.radians(self.x.max() / 2))

        # Get all allowed HKLs in range
        hkls = list(self.lattice.get_allowed_hkl(q_max))

        # Filter by q_min
        valid_hkls = []
        for hkl in hkls:
            q = self.lattice.GetQ(hkl)
            q_mag = np.linalg.norm(q)
            if q_mag >= q_min:
                valid_hkls.append((hkl, q_mag))

        if not valid_hkls:
            print(
                "Warning: No allowed reflections in the measured range. Skipping estimation."
            )
            return

        # Calculate Intensity for each
        # Energy in eV for Structure Factor
        en = 12398.4 / self.wavelength_val

        # Group by d-spacing (or Q) to handle multiplicity
        # Key: d_spacing (rounded), Value: sum of intensity
        grouped_intensities = {}

        for hkl, q_mag in valid_hkls:
            # Structure Factor
            F = self.crystal.StructureFactor(self.lattice.GetQ(hkl), en=en)
            F_sq = np.abs(F) ** 2

            # LP Factor
            # sin(theta) = q * lambda / 4pi
            sin_theta = q_mag * self.wavelength_val / (4 * np.pi)
            theta = np.arcsin(sin_theta)
            cos_theta = np.cos(theta)
            cos_2theta = np.cos(2 * theta)

            # Standard LP factor for unpolarized beam
            lp = (1 + cos_2theta**2) / (sin_theta**2 * cos_theta)

            intensity = F_sq * lp

            # Grouping
            d = 2 * np.pi / q_mag
            d_key = round(d, 4)

            if d_key not in grouped_intensities:
                grouped_intensities[d_key] = {"intensity": 0.0, "hkl": hkl, "d": d}

            grouped_intensities[d_key]["intensity"] += intensity

        # Find max theoretical intensity
        best_group = max(grouped_intensities.values(), key=lambda x: x["intensity"])
        best_hkl = best_group["hkl"]
        best_d_calc = best_group["d"]

        print(
            f"Strongest theoretical reflection: {best_hkl} (d_calc={best_d_calc:.4f} A)"
        )

        # 3. Scale Lattice
        # We assume the relative lattice parameters are correct, just scaled wrong.
        scale_factor = d_obs / best_d_calc
        print(f"Scaling lattice parameters by factor: {scale_factor:.5f}")

        self.lattice.a *= scale_factor

        # Only set b and c if they are free parameters (e.g. not constrained by symmetry)
        try:
            self.lattice.b *= scale_factor
        except RuntimeError:
            pass

        try:
            self.lattice.c *= scale_factor
        except RuntimeError:
            pass

        # Re-generate reflections with new lattice
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

        h, k, l = hkl  # noqa: E741
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
            return 180.0  # Peak is outside the measurable 2theta range (sin(theta) > 1)

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
        params.add("V", value=0.0, min=-1.0, max=1.0)
        params.add("W", value=0.5, min=0.001, max=5.0)

        # Peak Shape (Pseudo-Voigt mixing)
        # eta = eta_0 + eta_1 * 2theta
        params.add("eta_0", value=0.5, min=0.0, max=1.0)
        params.add("eta_1", value=0.0, min=-0.01, max=0.01, vary=False)

        # Asymmetry
        params.add("asymmetry", value=0.0, min=-1, max=1)

        # Zero shift & Displacement
        # Increased bounds for zero_shift to handle larger offsets
        params.add("zero_shift", value=0.0, min=-0.5, max=0.5)
        params.add("displacement", value=0.0, min=-0.5, max=0.5, vary=False)

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

            # Use the split_pseudo_voigt function from peak_fitting module
            # This avoids code duplication and ensures consistency

            # Optimization: only calculate near peak
            dx = self.x - two_theta
            mask = np.abs(dx) < fwhm * 10
            if not np.any(mask):
                continue

            # Calculate peak shape using the imported function
            # Note: split_pseudo_voigt takes (x, amplitude, center, sigma_l, sigma_r, fraction)
            # Here amplitude=1.0 because we multiply by intensity later
            peak_shape = split_pseudo_voigt(
                self.x[mask],
                amplitude=1.0,
                center=two_theta,
                sigma_l=sigma_l,
                sigma_r=sigma_r,
                fraction=eta,
            )

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

    def plot_result(self, result, save_path="data/wppf_result.png"):
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
        plt.title("Fit Result")
        plt.savefig(save_path)
        plt.close()
