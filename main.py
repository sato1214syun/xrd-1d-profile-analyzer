from pathlib import Path
from src.peak_fitting import find_and_fit_peaks, plot_multi_peak_fit
from src.peak_models import SplitPseudoVoigtModel
from src.ras_loader import load_ras_file

if __name__ == "__main__":
    # RASファイルのパスを指定
    ras_path = Path(
        r"data\XRD_profiles\XRD\250915_XRD_PB_ID_43_44\ID43_PB_合金Tiシード10nm_2000W.ras"
    )

    ras_df, _ = load_ras_file(ras_path)

    # GaussianModelで試す（B-splineバックグラウンド込み）

    ras_x = ras_df["x"].to_numpy()
    ras_y = ras_df["cps"].to_numpy()

    print("=== Gaussianモデルで試行（B-spline BG） ===")
    result_gaussian, peak_indices_gaussian = find_and_fit_peaks(
        ras_x,
        ras_y,
        SplitPseudoVoigtModel,
        height=None,
        prominence=None,
        distance=10,
        width=3,
        background_type="spline",  # B-spline
        num_knots=10,
    )

    if result_gaussian is not None:
        plot_multi_peak_fit(ras_x, ras_y, result_gaussian, peak_indices_gaussian)
