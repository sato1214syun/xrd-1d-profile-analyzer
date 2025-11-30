import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ras_loader import load_ras_file
from le_bail import LeBailFitter


def main():
    # Load Data
    ras_path = r"data\XRD_profiles\XRD\250915_XRD_PB_ID_43_44\ID44_PB_基板加熱100度_高真空_2000W.ras"
    cif_path = "data/cif/1538066.cif"  # NbTi

    print(f"Loading RAS: {ras_path}")
    df = load_ras_file(ras_path)
    x = df["x"].to_numpy()
    y = df["cps"].to_numpy()

    # Limit range for speed/testing
    mask = (x >= 30) & (x <= 90)
    x = x[mask]
    y = y[mask]

    print(f"Data points: {len(x)}")
    print(f"Range: {x.min():.2f} - {x.max():.2f}")

    # Initialize Fitter
    fitter = LeBailFitter(x, y, cif_path, wavelength="CuKa12", background_type="spline")

    # Fit
    print("Starting Fit...")
    result = fitter.fit(max_nfev=2000)

    print("Fit Complete.")
    print(f"Chi2: {result.chisqr:.2f}")
    print(f"RedChi: {result.redchi:.2f}")

    if hasattr(result, "r_wp"):
        print(f"R_wp: {result.r_wp:.2f}%")
        print(f"R_p:  {result.r_p:.2f}%")
        print(f"R_exp:{result.r_exp:.2f}%")
        print(f"GoF:  {result.gof:.2f}")

    # Print Lattice Params
    print("\nLattice Parameters:")
    for p in ["a", "b", "c", "alpha", "beta", "gamma"]:
        if p in result.params:
            print(
                f"{p}: {result.params[p].value:.4f} +/- {result.params[p].stderr or 0:.4f}"
            )

    # Print Profile Params
    print("\nProfile Parameters:")
    for p in [
        "U",
        "V",
        "W",
        "eta_0",
        "eta_1",
        "asymmetry",
        "zero_shift",
        "displacement",
    ]:
        if p in result.params:
            print(
                f"{p}: {result.params[p].value:.6f} +/- {result.params[p].stderr or 0:.6f}"
            )

    # Print Peak Residuals
    if hasattr(result, "peak_stats"):
        print("\nPeak Residuals:")
        print(
            f"{'HKL':<15} {'2Theta':<10} {'Obs':<10} {'Calc':<10} {'Resid':<10} {'Rel(%)':<10}"
        )
        print("-" * 70)
        for stat in result.peak_stats:
            hkl_str = str(tuple(stat["hkl"]))
            print(
                f"{hkl_str:<15} {stat['2theta']:.2f}      {stat['y_obs']:.1f}      {stat['y_calc']:.1f}      {stat['residual']:.1f}      {stat['rel_residual_percent']:.1f}%"
            )

    # Plot
    fitter.plot_result(result)


if __name__ == "__main__":
    main()
