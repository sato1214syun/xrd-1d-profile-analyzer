import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, PolynomialModel, SplineModel
from scipy.signal import find_peaks


def estimate_background_iterative(x, y, iterations=10, smoothing_window=11):
    """
    イテレーティブな方法でバックグラウンドを推定する
    （ピークを削りながら平滑化を繰り返す）

    Parameters
    ----------
    x : array_like
        x軸のデータ
    y : array_like
        y軸のデータ
    iterations : int
        イテレーション回数
    smoothing_window : int
        平滑化ウィンドウサイズ（奇数）

    Returns
    -------
    bg : array_like
        推定されたバックグラウンド
    """
    from scipy.signal import savgol_filter

    bg = np.copy(y)
    for _ in range(iterations):
        # 平滑化
        if smoothing_window > 3:
            bg = savgol_filter(bg, smoothing_window, 2)

        # 元のデータより大きい部分を元のデータで置き換える（ピークを削る）
        bg = np.minimum(bg, y)

    return bg


def find_and_fit_peaks(
    x,
    y,
    model_class,
    height=None,
    threshold=None,
    prominence=None,
    distance=None,
    width=None,
    background_type="spline",
    poly_degree=2,
    num_knots=10,
    delta=3.0,
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
    threshold : float, optional
        隣接するサンプルに対する垂直方向の距離の閾値
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
    delta : float, optional
        ピーク検出の閾値係数（δ値）。
        指定された場合、バックグラウンドを推定して差し引いたデータの標準偏差(σ)を計算し、
        height = delta * σ 以上のピークを検出する。

    Returns
    -------
    result : ModelResult
        フィッティング結果（ピーク + バックグラウンド）
    peak_indices : array
        検出されたピークのインデックス
    """
    # バックグラウンドを推定して除去（ピーク検出用）
    bg_estimated = estimate_background_iterative(x, y)
    detection_y = y - bg_estimated

    # デフォルト値の設定
    if height is None and delta is None:
        height = (
            detection_y.max() - detection_y.min()
        ) * 0.1  # バックグラウンド除去後のデータ範囲の10%
    if prominence is None and delta is None:
        prominence = (
            detection_y.max() - detection_y.min()
        ) * 0.05  # バックグラウンド除去後のデータ範囲の5%

    # ピーク検出用の閾値を設定
    detection_height = height

    # deltaが指定された場合、標準偏差に基づいて閾値を設定
    if delta is not None:
        # 標準偏差を計算して閾値を設定
        sigma = np.std(detection_y)
        detection_height = delta * sigma

        print(f"δ値 ({delta}) を用いてピーク検出を行います（バックグラウンド除去後）")
        print(f"  バックグラウンド除去後の標準偏差(σ): {sigma:.4f}")
        print(f"  検出閾値 (δ * σ): {detection_height:.4f}")

        # prominenceはdeltaが指定されている場合はデフォルトでNoneにする
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

    print(f"\n検出されたピーク数: {len(peak_indices)}")
    print(f"ピーク位置 (2θ): {x[peak_indices]}")

    if len(peak_indices) == 0:
        print("ピークが検出されませんでした。パラメータを調整してください。")
        return None, peak_indices

    # バックグラウンドモデルを作成
    print(f"\nバックグラウンドモデル: {background_type}")
    if background_type == "spline":
        # B-splineモデル（推奨）
        # ノット位置を均等に配置
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

    # 複合モデルを構築（バックグラウンドから開始）
    composite_model = background_model
    params = bg_params

    # 各ピークモデルを追加
    for i, peak_idx in enumerate(peak_indices):
        peak_model = model_class(prefix=f"p{i}_")
        composite_model += peak_model

        center = x[peak_idx]
        amplitude = y[peak_idx] - y.min()

        # ピーク幅を推定
        half_height = y.min() + amplitude / 2
        left_idx = peak_idx
        right_idx = peak_idx
        while left_idx > 0 and y[left_idx] > half_height:
            left_idx -= 1
        while right_idx < len(y) - 1 and y[right_idx] > half_height:
            right_idx += 1
        fwhm = abs(x[right_idx] - x[left_idx])
        if fwhm < 0.01:  # FWHMが小さすぎる場合のフォールバック
            fwhm = 0.1
        sigma = fwhm / 2.355

        params.update(peak_model.make_params())
        params[f"p{i}_amplitude"].set(value=amplitude, min=0, max=amplitude * 10)
        params[f"p{i}_center"].set(value=center, min=center - fwhm, max=center + fwhm)
        params[f"p{i}_sigma"].set(value=sigma, min=sigma * 0.1, max=sigma * 10)
        if "fraction" in peak_model.param_names:
            params[f"p{i}_fraction"].set(value=0.5, min=0, max=1)
        if "skew" in peak_model.param_names:
            params[f"p{i}_skew"].set(value=0.0, min=-5, max=5)

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

    # ピーク位置にマーカー
    ax1.plot(
        x[peak_indices],
        y[peak_indices],
        "rv",
        markersize=10,
        label="Detected Peaks",
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
    for i in range(len(peak_indices)):
        print(f"\nPeak {i + 1}:")
        prefix = f"p{i}_"
        for param_name in ["amplitude", "center", "sigma", "fwhm", "fraction", "skew"]:
            full_name = f"{prefix}{param_name}"
            if full_name in result.params:
                param = result.params[full_name]
                stderr = param.stderr if param.stderr else 0
                print(f"  {param_name}: {param.value:.4f} ± {stderr:.4f}")

    return result
