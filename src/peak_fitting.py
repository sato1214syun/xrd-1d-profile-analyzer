import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.models import LinearModel, PolynomialModel, SplineModel
from scipy.signal import find_peaks
import polars as pl

try:
    from .peak_models import SplitPseudoVoigtModel, split_pseudo_voigt
except ImportError:
    from peak_models import SplitPseudoVoigtModel, split_pseudo_voigt


def create_background_model(
    x, y, background_type="spline", poly_degree=2, num_knots=10
):
    """
    バックグラウンドモデルを作成し、初期パラメータを推定する
    """
    print(f"\nバックグラウンドモデル: {background_type}")
    if background_type == "spline":
        knot_positions = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]
        background_model = SplineModel(prefix="bg_", xknots=knot_positions)
        bg_params = background_model.guess(y, x=x)
        print(f"  B-splineノット数: {num_knots}")
    elif background_type == "polynomial":
        background_model = PolynomialModel(degree=poly_degree, prefix="bg_")
        bg_params = background_model.guess(y, x=x)
        print(f"  多項式次数: {poly_degree}")
    else:  # 'linear'
        background_model = LinearModel(prefix="bg_")
        bg_params = background_model.make_params()
        bg_params["bg_slope"].set(value=0)
        bg_params["bg_intercept"].set(value=y.min())

    return background_model, bg_params


def setup_peak_models(
    model_class, peak_indices, properties, x, y, background_model, bg_params
):
    """
    検出されたピークに基づいて複合モデルを構築し、パラメータを初期化する
    """
    composite_model = background_model
    params = bg_params.copy()

    # インデックス配列を作成（補間用）
    indices_arr = np.arange(len(x))

    # FWHMを一括計算（可能な場合）
    fwhms = None
    if "left_ips" in properties and "right_ips" in properties:
        x_lefts = np.interp(properties["left_ips"], indices_arr, x)
        x_rights = np.interp(properties["right_ips"], indices_arr, x)
        fwhms = np.abs(x_rights - x_lefts)

    for i, peak_idx in enumerate(peak_indices):
        peak_model = model_class(prefix=f"p{i}_")
        composite_model += peak_model

        center = x[peak_idx]
        # propertiesにprominencesがあればそれを使う、なければデータから推定
        amplitude = (
            properties["prominences"][i]
            if "prominences" in properties
            else y[peak_idx] - y.min()
        )

        # ピーク幅を推定
        if fwhms is not None:
            fwhm = fwhms[i]
        else:
            # 従来の手動探索（バックアップ）
            half_height = y.min() + amplitude / 2

            # 左側探索: peak_idxより前でhalf_height以下の最後のインデックス
            left_mask = y[:peak_idx] <= half_height
            left_indices = np.where(left_mask)[0]
            left_idx = left_indices[-1] if left_indices.size > 0 else 0

            # 右側探索: peak_idxより後でhalf_height以下の最初のインデックス
            right_mask = y[peak_idx:] <= half_height
            right_indices = np.where(right_mask)[0]
            right_idx = (
                peak_idx + right_indices[0] if right_indices.size > 0 else len(y) - 1
            )

            fwhm = abs(x[right_idx] - x[left_idx])

        sigma = fwhm / 2.355
        hwhm = fwhm / 2.0

        params.update(peak_model.make_params())
        params[f"p{i}_amplitude"].set(
            value=amplitude, min=amplitude / 2, max=amplitude * 1.5
        )
        params[f"p{i}_center"].set(value=center, min=center - hwhm, max=center + hwhm)

        # Sigma関連のパラメータを一括設定
        for suffix, val in [("sigma", sigma), ("sigma_l", hwhm), ("sigma_r", hwhm)]:
            key = f"p{i}_{suffix}"
            if key in params:
                params[key].set(value=val, min=val * 0.1, max=val * 3)

        # その他のオプションパラメータ設定
        optional_params = {"fraction": (0.5, 0, 1), "skew": (0.0, -5, 5)}
        for name, (val, min_val, max_val) in optional_params.items():
            if name in peak_model.param_names:
                params[f"p{i}_{name}"].set(value=val, min=min_val, max=max_val)

    return composite_model, params


def estimate_background_iterative(
    x, y, background_type="spline", poly_degree=2, num_knots=10, iterations=10
):
    """
    指定されたモデルを用いてイテレーティブにバックグラウンドを推定する
    （ピークを削りながらフィッティングを繰り返す）

    Parameters
    ----------
    x : array_like
        x軸のデータ
    y : array_like
        y軸のデータ
    background_type : str
        バックグラウンドモデルのタイプ
    poly_degree : int
        多項式の次数
    num_knots : int
        スプラインのノット数
    iterations : int
        イテレーション回数

    Returns
    -------
    bg : array_like
        推定されたバックグラウンド
    """
    # モデルと初期パラメータを取得
    # 注: ここでprint出力が出るが、ログとして許容する
    model, params = create_background_model(
        x, y, background_type, poly_degree, num_knots
    )

    y_work = np.copy(y)
    for _ in range(iterations):
        # フィッティング
        result = model.fit(y_work, params, x=x)
        bg_fit = result.best_fit

        # 元のデータyとフィッティング結果bg_fitの小さい方を採用
        # これにより、ピーク部分（y > bg_fit となる部分）が削られていく
        y_work = np.minimum(y, bg_fit)

        # パラメータ更新
        params = result.params

    # 最終的なバックグラウンド
    return model.fit(y_work, params, x=x).best_fit


def find_and_fit_peaks(
    x,
    y,
    model_class,
    height=None,
    prominence=None,
    distance=None,
    width=None,
    background_type="spline",
    poly_degree=2,
    num_knots=10,
    delta=None,
):
    """
    複数のピークを自動検出し、バックグラウンドも含めて同時フィッティングする

    Parameters
    ----------
    x : array_like
        x軸のデータ
    y : array_like
        y軸のデータ
    model_class : class
        使用するモデルクラス (例: GaussianModel, PseudoVoigtModel)
    height : float, optional
        ピーク検出の最小高さ
    prominence : float, optional
        ピークの突出度（周囲との相対的な高さ）
    distance : int, optional
        ピーク間の最小距離（データポイント数）
    width : int or tuple, optional
        ピークの最小幅（データポイント数）
    background_type : str, optional
        バックグラウンドモデルのタイプ ('spline', 'polynomial', 'linear')
        デフォルト: 'spline' (B-spline)
    poly_degree : int, optional
        多項式バックグラウンドの次数（background_type='polynomial'の場合）
    num_knots : int, optional
        B-splineのノット数（background_type='spline'の場合）
        デフォルト: 10

    Returns
    -------
    result : ModelResult
        フィッティング結果（ピーク + バックグラウンド）
    peak_indices : array
        検出されたピークのインデックス
    """

    # ピークを検出
    peak_indices, properties = detect_peaks_from_data(
        x,
        y,
        height=height,
        prominence=prominence,
        distance=distance,
        width=width,
        delta=delta,
        background_type=background_type,
        poly_degree=poly_degree,
        num_knots=num_knots,
    )

    if len(peak_indices) == 0:
        print("ピークが検出されませんでした。パラメータを調整してください。")
        return None, peak_indices

    # バックグラウンドモデルを作成
    background_model, bg_params = create_background_model(
        x,
        y,
        background_type=background_type,
        poly_degree=poly_degree,
        num_knots=num_knots,
    )

    # 複合モデルを構築（バックグラウンドから開始）
    composite_model, params = setup_peak_models(
        model_class, peak_indices, properties, x, y, background_model, bg_params
    )

    # 初期パラメータでモデル評価をテスト
    print("\n=== Testing initial parameters ===")
    try:
        init_y = composite_model.eval(params, x=x)
        if np.any(np.isnan(init_y)) or np.any(np.isinf(init_y)):
            print("警告: 初期パラメータでNaN/Infが発生しています")
            return None, peak_indices
        else:
            print("初期パラメータは正常です")
    except Exception as e:
        print(f"初期評価でエラー: {e}")
        return None, peak_indices

    # フィッティング実行
    print("\n=== フィッティング開始 ===")
    try:
        result = composite_model.fit(y, params, x=x, nan_policy="omit", max_nfev=5000)
        print(f"フィッティング成功: {result.success}")
        if not result.success:
            print(f"警告: {result.message}")
    except Exception as e:
        print(f"フィッティングエラー: {e}")
        return None, peak_indices

    return result, peak_indices


def detect_peaks_from_data(
    x,
    y,
    height=None,
    threshold=None,
    prominence=None,
    distance=None,
    width=None,
    delta=None,
    background_type="spline",
    poly_degree=2,
    num_knots=10,
):
    """
    データからピークを検出する（バックグラウンド除去オプション付き）
    """
    # バックグラウンドを推定して除去（ピーク検出用）
    bg_estimated = estimate_background_iterative(
        x,
        y,
        background_type=background_type,
        poly_degree=poly_degree,
        num_knots=num_knots,
    )
    detection_y = y - bg_estimated

    # デフォルト値の設定
    if height is None and delta is None:
        height = (detection_y.max() - detection_y.min()) * 0.1
    if prominence is None and delta is None:
        prominence = (detection_y.max() - detection_y.min()) * 0.05

    # ピーク検出用の閾値を設定
    detection_height = height

    # deltaが指定された場合、標準偏差に基づいて閾値を設定
    if delta is not None:
        sigma = np.std(detection_y)
        detection_height = delta * sigma
        print(f"δ値 ({delta}) を用いてピーク検出を行います（バックグラウンド除去後）")
        print(f"  バックグラウンド除去後の標準偏差(σ): {sigma:.4f}")
        print(f"  検出閾値 (δ * σ): {detection_height:.4f}")

        if prominence is None:
            prominence = None

    # ピークを検出
    peak_indices, properties = find_peaks(
        detection_y,
        height=detection_height,
        threshold=threshold,
        prominence=prominence,
        distance=distance,
        width=width,
    )

    return peak_indices, properties


def plot_multi_peak_fit(x, y, result, peak_indices):
    """
    複数ピークのフィッティング結果をプロット

    Parameters
    ----------
    x : array_like
        x軸のデータ
    y : array_like
        y軸のデータ
    result : ModelResult
        フィッティング結果
    peak_indices : array
        検出されたピークのインデックス
    """

    # フィッティング統計を表示
    print("\n=== Fitting Statistics ===")
    print(f"Chi-squared: {result.chisqr:.4e}")
    print(f"Reduced chi-squared: {result.redchi:.4e}")
    print(f"R-squared: {result.rsquared:.6f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")

    # バックグラウンドパラメータを表示
    print("\n=== Background Parameters ===")
    for param_name in result.params:
        if param_name.startswith("bg_"):
            param = result.params[param_name]
            stderr = param.stderr if param.stderr else 0
            print(f"{param_name}: {param.value:.4f} ± {stderr:.4f}")

    # 各ピークのパラメータを表示
    print("\n=== Peak Parameters ===")

    peak_data = []
    for i in range(len(peak_indices)):
        prefix = f"p{i}_"

        # パラメータ取得用ヘルパー関数
        def get_val_err(name):
            full_name = f"{prefix}{name}"
            if full_name in result.params:
                param = result.params[full_name]
                val = param.value
                err = param.stderr if param.stderr is not None else 0.0
                return val, err
            return None, None

        center, center_err = get_val_err("center")
        amplitude, amplitude_err = get_val_err("amplitude")
        fwhm, fwhm_err = get_val_err("fwhm")
        fraction, fraction_err = get_val_err("fraction")
        skew, skew_err = get_val_err("skew")
        sigma_l, sigma_l_err = get_val_err("sigma_l")
        sigma_r, sigma_r_err = get_val_err("sigma_r")

        # fwhmが取得できない場合、sigma_lとsigma_rから計算
        if fwhm is None and sigma_l is not None and sigma_r is not None:
            fwhm = sigma_l + sigma_r
            # 誤差伝播 (簡易的: 二乗和の平方根)
            if sigma_l_err is not None and sigma_r_err is not None:
                fwhm_err = np.sqrt(sigma_l_err**2 + sigma_r_err**2)
            else:
                fwhm_err = None

        peak_data.append(
            {
                "center[deg]": center,
                "center_σ[deg]": center_err,
                "amplitude[cps]": amplitude,
                "amplitude_σ[cps]": amplitude_err,
                "fwmh[deg]": fwhm,
                "fwmh_σ[deg]": fwhm_err,
                "fraction": fraction,
                "fraction_σ": fraction_err,
                "sigma_l": sigma_l,
                "sigma_l_σ": sigma_l_err,
                "sigma_r": sigma_r,
                "sigma_r_σ": sigma_r_err,
            }
        )

    if peak_data:
        df = pl.DataFrame(peak_data)
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 上段: データとフィッティング結果
    ax1.plot(x, y, "b.", label="Data", markersize=3)
    ax1.plot(x, result.best_fit, "r-", label="Best Fit", linewidth=2)

    # 各ピークのコンポーネントをプロット
    comps = result.eval_components(x=x)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(peak_indices)))
    for i, color in enumerate(colors):
        if f"p{i}_" in comps:
            ax1.plot(
                x,
                comps[f"p{i}_"],
                "--",
                color=color,
                label=f"Peak {i + 1} (2θ={x[peak_indices[i]]:.2f}°)",
                linewidth=1.5,
            )

    # バックグラウンド
    if "bg_" in comps:
        ax1.plot(x, comps["bg_"], "k:", label="Background", linewidth=2)

    # フィッティングしたピーク位置にマーカー
    if peak_data:
        fitted_centers = [
            d["center[deg]"] for d in peak_data if d["center[deg]"] is not None
        ]
        if fitted_centers:
            fitted_heights = result.eval(x=np.array(fitted_centers))
            ax1.plot(
                fitted_centers,
                fitted_heights,
                "rv",
                markersize=10,
                label="Fitted Peaks",
                zorder=5,
            )

    ax1.set_xlabel("2θ (degrees)")
    ax1.set_ylabel("Intensity (cps)")
    ax1.set_title("Multi-Peak Fitting with Background")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下段: 残差
    residual = y - result.best_fit
    ax2.plot(x, residual, "g-", linewidth=1)
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=1)
    ax2.set_xlabel("2θ (degrees)")
    ax2.set_ylabel("Residual")
    ax2.set_title("Fit Residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return result
