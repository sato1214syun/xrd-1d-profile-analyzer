import streamlit as st
import plotly.graph_objects as go
import polars as pl
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ras_loader import load_ras_file
from peak_fitting import find_and_fit_peaks
from peak_models import SplitPseudoVoigtModel
from wppf.pawley import PawleyFitter
from wppf.le_bail import LeBailFitter


def main():
    st.set_page_config(layout="wide", page_title="XRD Profile Analyzer")

    st.title("XRD Profile Analyzer")

    # Sidebar for inputs
    st.sidebar.header("Settings")

    # 1. RAS File Selection
    uploaded_file = st.sidebar.file_uploader("Upload RAS File", type=["ras"])

    # CIF Selection
    cif_dir = Path("data/cif")
    if not cif_dir.exists():
        st.error(f"CIF directory not found: {cif_dir}")
        return

    cif_files = list(cif_dir.glob("*.cif"))
    selected_cif_path = st.sidebar.selectbox(
        "Select CIF File (for WPPF)", cif_files, format_func=lambda x: x.name
    )

    if uploaded_file:
        # Save uploaded file to a temporary file because ras_loader expects a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ras") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Load Data
            df = load_ras_file(tmp_path)
            x = df["x"].to_numpy()
            y = df["cps"].to_numpy()  # Using cps for fitting

            # 2. Peak Fitting
            st.subheader("Peak Fitting Analysis")

            with st.spinner("Running Peak Fitting..."):
                # Using default parameters for now
                result, indices = find_and_fit_peaks(
                    x,
                    y,
                    model_class=SplitPseudoVoigtModel,
                    background_type="spline",
                    num_knots=20,
                )

            # 3. Plotly Visualization
            fig = go.Figure()

            # Raw Data
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name="Observed",
                    marker=dict(size=3, color="black", opacity=0.5),
                )
            )

            # Best Fit
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=result.best_fit,
                    mode="lines",
                    name="Best Fit",
                    line=dict(color="red", width=1.5),
                )
            )

            # Background
            comps = result.eval_components()
            if "background" in comps:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=comps["background"],
                        mode="lines",
                        name="Background",
                        line=dict(color="blue", dash="dash"),
                    )
                )

            fig.update_layout(
                title=f"Peak Fitting Result: {uploaded_file.name}",
                xaxis_title="2Theta (deg)",
                yaxis_title="Intensity (cps)",
                height=600,
                legend=dict(x=0.8, y=1),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display Peak Parameters
            st.markdown("### Peak Parameters")
            peak_params = []

            # Parse lmfit parameters to extract peak info
            # Assuming prefix format like "p0_", "p1_" etc.
            # We look for 'center', 'fwhm', 'height' (or amplitude)

            # Group parameters by prefix
            prefixes = set()
            for name in result.params:
                if "_" in name:
                    prefix = name.split("_")[0]
                    if prefix.startswith("p") and prefix[1:].isdigit():
                        prefixes.add(prefix)

            sorted_prefixes = sorted(list(prefixes), key=lambda p: int(p[1:]))

            for prefix in sorted_prefixes:
                center = result.params.get(f"{prefix}_center")
                fwhm = result.params.get(f"{prefix}_fwhm")
                height = result.params.get(f"{prefix}_height")
                amplitude = result.params.get(f"{prefix}_amplitude")

                if center:
                    peak_params.append(
                        {
                            "Peak ID": prefix,
                            "Center (deg)": center.value,
                            "FWHM (deg)": fwhm.value if fwhm else np.nan,
                            "Height": height.value if height else np.nan,
                            "Amplitude": amplitude.value if amplitude else np.nan,
                        }
                    )

            if peak_params:
                df_peaks = pl.DataFrame(peak_params)
                st.dataframe(df_peaks, use_container_width=True)
            else:
                st.info("No peaks detected or parameters could not be parsed.")

            # 4. WPPF Controls
            st.divider()
            st.header("Whole Powder Pattern Fitting (WPPF)")

            col1, col2 = st.columns([1, 2])

            with col1:
                method = st.radio("Fitting Method", ["Pawley", "Le Bail"])
                cycles = st.number_input("Cycles", min_value=1, max_value=100, value=20)

            with col2:
                st.write(f"**Selected Crystal Structure:** {selected_cif_path.name}")
                run_wppf = st.button("Run WPPF Analysis", type="primary")

            # 5. WPPF Execution & Results
            if run_wppf:
                run_wppf_analysis(x, y, selected_cif_path, method, cycles)

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@st.dialog("WPPF Results", width="large")
def run_wppf_analysis(x, y, cif_path, method, cycles):
    st.write(f"Running {method} refinement on {cif_path.name}...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Instantiate Fitter
        if method == "Pawley":
            fitter = PawleyFitter(x, y, cif_path=cif_path)
        else:
            fitter = LeBailFitter(x, y, cif_path=cif_path)

        status_text.text("Fitting in progress...")

        # Run Fit
        # Note: The current fit() method doesn't support a progress callback easily
        # without modifying the class, so we just run it.
        result = fitter.fit(cycles=cycles)

        progress_bar.progress(100)
        status_text.text("Fitting Complete!")

        # Display Statistics
        st.subheader("Fit Statistics")
        stats_cols = st.columns(4)
        stats_cols[0].metric("R_wp", f"{result.r_wp:.2f}%")
        stats_cols[1].metric("R_p", f"{result.r_p:.2f}%")
        stats_cols[2].metric("Chi2", f"{result.chisqr:.2f}")
        stats_cols[3].metric("GoF", f"{result.gof:.2f}")

        # 6. Results DataFrame (Lattice & Peaks)

        # Lattice Parameters
        st.subheader("Lattice Parameters")
        lattice_params = {
            "Parameter": ["a", "b", "c", "alpha", "beta", "gamma"],
            "Value": [
                result.params["a"].value,
                result.params["b"].value,
                result.params["c"].value,
                result.params["alpha"].value,
                result.params["beta"].value,
                result.params["gamma"].value,
            ],
            "Stderr": [
                result.params["a"].stderr,
                result.params["b"].stderr,
                result.params["c"].stderr,
                result.params["alpha"].stderr,
                result.params["beta"].stderr,
                result.params["gamma"].stderr,
            ],
        }
        df_lattice = pl.DataFrame(lattice_params)
        st.dataframe(df_lattice)

        # Peak Stats
        st.subheader("Peak Reflections")
        if hasattr(result, "peak_stats") and result.peak_stats:
            # Flatten the peak stats list of dicts
            # Each item in peak_stats is like:
            # {'hkl': (1, 1, 0), '2theta': 38.5, 'd': 2.3, 'fwhm': 0.1, 'intensity': 1000}

            # Convert tuple HKL to string for display
            formatted_stats = []
            for p in result.peak_stats:
                p_copy = p.copy()
                p_copy["hkl"] = str(p["hkl"])
                formatted_stats.append(p_copy)

            df_peaks_wppf = pl.DataFrame(formatted_stats)
            st.dataframe(df_peaks_wppf)

        # Plot Result
        st.subheader("Fitted Profile")
        fig_wppf = go.Figure()
        fig_wppf.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Observed",
                marker=dict(size=2, color="black"),
            )
        )
        fig_wppf.add_trace(
            go.Scatter(
                x=x,
                y=result.best_fit,
                mode="lines",
                name="Calculated",
                line=dict(color="red"),
            )
        )
        fig_wppf.add_trace(
            go.Scatter(
                x=x,
                y=y - result.best_fit,
                mode="lines",
                name="Difference",
                line=dict(color="gray"),
            )
        )

        st.plotly_chart(fig_wppf, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during WPPF: {e}")
        # Print traceback to console for debugging
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
