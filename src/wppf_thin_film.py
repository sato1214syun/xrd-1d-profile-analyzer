# 薄膜試料向け改善版WPPF実装

import numpy as np
import polars as pl
from lmfit import Parameters
from lmfit.models import LinearModel, SplineModel
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def wppf_hybrid_thin_film(
    x,
    y,
    theoretical_peaks_df,
    model_class,
    background_type="spline",
    num_knots=10,
    peak_matching_tolerance=0.5,
    min_peak_height=None,
    prominence_factor=0.03,
):
    """
    アプローチ1: ハイブリッド法（推奨）
    実験データで自動検出したピークと理論ピークをマッチングさせる

    Parameters
    ----------
    x : array_like
        実験データのx軸（2θ）
    y : array_like
        実験データの強度
    theoretical_peaks_df : polars.DataFrame
        理論ピーク情報（two_theta列が必要）
    model_class : class
        使用するピークモデル
    background_type : str
        バックグラウンドモデル
    num_knots : int
        B-splineノット数
    peak_matching_tolerance : float
        理論ピークとのマッチング許容範囲（度）
    min_peak_height : float, optional
        ピーク検出の最小高さ
    prominence_factor : float
        突出度の係数

    Returns
    -------
    result : ModelResult
        フィッティング結果（matched_peaks, unmatched_peaks属性付き）
    """
    print("=== Hybrid WPPF for Thin Films ===")

    # 実験データでピーク検出
    if min_peak_height is None:
        min_peak_height = (y.max() - y.min()) * 0.05

    exp_peak_indices, _ = find_peaks(
        y,
        height=min_peak_height,
        prominence=(y.max() - y.min()) * prominence_factor,
        distance=10,
    )

    exp_peak_positions = x[exp_peak_indices]
    print(f"実験データで検出されたピーク数: {len(exp_peak_positions)}")
    print(f"ピーク位置: {exp_peak_positions}")

    # 理論ピークとマッチング
    matched_peaks = []
    unmatched_exp_peaks = []
    used_theory_indices = set()

    for exp_idx, exp_pos in zip(exp_peak_indices, exp_peak_positions):
        # 最も近い理論ピークを探す
        closest_theory = None
        min_distance = float("inf")
        closest_theory_idx = None

        for idx, row in enumerate(theoretical_peaks_df.iter_rows(named=True)):
            if idx in used_theory_indices:
                continue
            theory_pos = row["two_theta"]
            distance = abs(exp_pos - theory_pos)
            if distance < min_distance and distance < peak_matching_tolerance:
                min_distance = distance
                closest_theory = row
                closest_theory_idx = idx

        if closest_theory is not None:
            used_theory_indices.add(closest_theory_idx)
            matched_peaks.append(
                {
                    "exp_position": exp_pos,
                    "exp_index": exp_idx,
                    "theory_position": closest_theory["two_theta"],
                    "theory_intensity": closest_theory.get("relative_intensity", 100),
                    "d_spacing": closest_theory.get("d_spacing", 0),
                    "matched": True,
                }
            )
        else:
            # 理論ピークとマッチしない実験ピーク
            unmatched_exp_peaks.append(
                {"exp_position": exp_pos, "exp_index": exp_idx, "matched": False}
            )

    print(f"\n理論ピークとマッチしたピーク数: {len(matched_peaks)}")
    print(f"マッチしなかったピーク数: {len(unmatched_exp_peaks)}")

    if len(matched_peaks) == 0:
        print("警告: 理論ピークと一致する実験ピークが見つかりませんでした")
        print(
            f"peak_matching_toleranceを大きくするか（現在{peak_matching_tolerance}°）、"
        )
        print("データ範囲を確認してください")
        return None

    # バックグラウンドモデル
    if background_type == "spline":
        knot_positions = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]
        background_model = SplineModel(prefix="bg_", xknots=knot_positions)
        params = background_model.guess(y, x=x)
    else:
        background_model = LinearModel(prefix="bg_")
        params = background_model.make_params()
        params["bg_slope"].set(value=0)
        params["bg_intercept"].set(value=y.min())

    # WPPFパラメータの追加 (Cagliotiパラメータとゼロシフト、格子歪み)
    # FWHM^2 = U * tan(theta)^2 + V * tan(theta) + W
    # theta = 2theta / 2
    params.add("U", value=0.01, min=0, max=1.0)
    params.add("V", value=0.01, min=-1.0, max=1.0)
    params.add("W", value=0.01, min=0, max=1.0)
    params.add("zero_shift", value=0.0, min=-1.0, max=1.0)
    params.add("lattice_strain", value=0.0, min=-0.05, max=0.05)

    composite_model = background_model

    # 理論ピークのモデル構築 (Pawley法ライクなアプローチ)
    # 範囲内のすべての理論ピークを使用
    x_min, x_max = x.min(), x.max()
    target_peaks = theoretical_peaks_df.filter(
        (pl.col("two_theta") >= x_min) & (pl.col("two_theta") <= x_max)
    )

    print(f"フィッティング対象の理論ピーク数: {len(target_peaks)}")

    for i, row in enumerate(target_peaks.iter_rows(named=True)):
        peak_model = model_class(prefix=f"p{i}_")
        composite_model += peak_model

        theory_pos = row["two_theta"]

        # パラメータ設定
        params.update(peak_model.make_params())

        # 中心位置: 格子歪みとゼロシフトを考慮
        # 2theta_obs = 2 * arcsin( sin(theta_th) / (1 + strain) ) + zero_shift
        th_rad = theory_pos / 2 * np.pi / 180
        sin_th = np.sin(th_rad)

        # lmfitの式として記述 (piはlmfitの数式パーサで利用可能)
        center_expr = (
            f"2 * 180 / pi * arcsin({sin_th} / (1 + lattice_strain)) + zero_shift"
        )

        params[f"p{i}_center"].set(value=theory_pos, expr=center_expr)

        # 幅 (Sigma/FWHM): Caglioti式で拘束
        # sigma = FWHM / 2.355 (Gaussianの場合)
        # tan_theta = tan(center / 2 * pi / 180)
        # FWHM = sqrt(U * tan_theta**2 + V * tan_theta + W)

        # lmfitの式として記述
        # centerは変数なので、p{i}_centerを使う
        tan_theta_expr = f"tan(p{i}_center / 2 * pi / 180)"
        fwhm_expr = f"sqrt(U * {tan_theta_expr}**2 + V * {tan_theta_expr} + W)"
        sigma_expr = f"{fwhm_expr} / 2.355"

        if f"p{i}_sigma" in params:
            params[f"p{i}_sigma"].set(expr=sigma_expr)

        # SplitPseudoVoigtなどの場合
        if f"p{i}_sigma_l" in params:
            params[f"p{i}_sigma_l"].set(expr=f"{fwhm_expr} / 2")
        if f"p{i}_sigma_r" in params:
            params[f"p{i}_sigma_r"].set(expr=f"{fwhm_expr} / 2")

        # 強度: 初期値は実験データから推定、あるいは理論強度を使用
        # ここではPawley法のように自由パラメータとするが、初期値として理論強度を考慮
        # 近くに実験ピークがあればその高さを参照
        closest_exp_idx = (np.abs(x - theory_pos)).argmin()
        est_height = max(0, y[closest_exp_idx] - y.min())
        if est_height < 1e-3:
            est_height = (
                row.get("relative_intensity", 100) * (y.max() - y.min()) / 1000.0
            )

        params[f"p{i}_amplitude"].set(value=est_height, min=0)

        if "fraction" in peak_model.param_names:
            params[f"p{i}_fraction"].set(value=0.5, min=0, max=1)

    # マッチしなかった実験ピーク（不純物など）も追加
    # ただし、理論ピークと重複しないもののみ
    # 判定基準: どの理論ピークからも一定距離以上離れている
    extra_peak_count = 0
    for exp_idx, exp_pos in zip(exp_peak_indices, exp_peak_positions):
        # 理論ピークとの最小距離
        min_dist = target_peaks.select(
            (pl.col("two_theta") - exp_pos).abs().min()
        ).item()

        if min_dist is None or min_dist > peak_matching_tolerance:
            # 追加ピークとして扱う
            j = len(target_peaks) + extra_peak_count
            peak_model = model_class(prefix=f"p{j}_")
            composite_model += peak_model

            amplitude_guess = y[exp_idx] - y.min()

            params.update(peak_model.make_params())
            params[f"p{j}_amplitude"].set(value=amplitude_guess, min=0)
            params[f"p{j}_center"].set(
                value=exp_pos, min=exp_pos - 0.2, max=exp_pos + 0.2
            )

            # 追加ピークはCaglioti拘束を受けない（独立）
            params[f"p{j}_sigma"].set(value=0.1, min=0.01, max=1.0)

            if "fraction" in peak_model.param_names:
                params[f"p{j}_fraction"].set(value=0.5, min=0, max=1)

            extra_peak_count += 1

    print(f"追加された未同定ピーク数: {extra_peak_count}")

    total_peaks = len(target_peaks) + extra_peak_count
    print(f"総ピークモデル数: {total_peaks}")
    print(f"総ピークモデル数: {total_peaks}")

    # フィッティング実行
    print("\n=== フィッティング開始 ===")
    result = composite_model.fit(y, params, x=x, nan_policy="omit", max_nfev=10000)

    print(f"フィッティング成功: {result.success}")
    if not result.success:
        print(f"警告: {result.message}")

    # 結果にマッチング情報を追加
    result.matched_peaks = matched_peaks
    result.unmatched_peaks = unmatched_exp_peaks

    return result


def wppf_with_lattice_correction(
    x,
    y,
    theoretical_peaks_df,
    model_class,
    background_type="spline",
    num_knots=10,
    allow_lattice_strain=True,
    max_strain_percent=2.0,
):
    """
    アプローチ2: 格子定数補正付きWPPF
    応力による格子歪みを考慮（すべてのピークを同じ比率でシフト）

    Parameters
    ----------
    x : array_like
        実験データのx軸（2θ）
    y : array_like
        実験データの強度
    theoretical_peaks_df : polars.DataFrame
        理論ピーク情報
    model_class : class
        使用するピークモデル
    background_type : str
        バックグラウンドモデル
    num_knots : int
        B-splineノット数
    allow_lattice_strain : bool
        格子歪みパラメータを使用するか
    max_strain_percent : float
        最大格子歪み（%）

    Returns
    -------
    result : ModelResult
        フィッティング結果（lattice_strain値を含む）
    """
    print("=== WPPF with Lattice Parameter Correction ===")

    params = Parameters()

    # 格子歪みパラメータ（共通）
    if allow_lattice_strain:
        params.add(
            "lattice_strain",
            value=0.0,
            min=-max_strain_percent / 100,
            max=max_strain_percent / 100,
        )
    else:
        params.add("lattice_strain", value=0.0, vary=False)

    # バックグラウンド
    if background_type == "spline":
        knot_positions = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]
        background_model = SplineModel(prefix="bg_", xknots=knot_positions)
        bg_params = background_model.guess(y, x=x)
        params.update(bg_params)
    else:
        background_model = LinearModel(prefix="bg_")
        params.add("bg_slope", value=0)
        params.add("bg_intercept", value=y.min())

    composite_model = background_model

    # 各理論ピークに対してモデル構築
    for i, row in enumerate(theoretical_peaks_df.iter_rows(named=True)):
        peak_model = model_class(prefix=f"p{i}_")
        composite_model += peak_model

        theory_pos = row["two_theta"]

        # 実験データから初期振幅を推定
        idx = np.argmin(np.abs(x - theory_pos))
        amplitude_guess = max(y[idx] - y.min(), (y.max() - y.min()) * 0.05)

        fwhm_guess = 0.2
        sigma = fwhm_guess / 2.355

        # ピーク位置を格子歪みパラメータで制御
        # Bragg式より: d = λ/(2sinθ), 歪みがある場合 d' = d/(1+ε)
        # 簡易的に: 2θ' ≈ 2θ * (1 - ε) (小角近似)
        params.add(
            f"p{i}_amplitude", value=amplitude_guess, min=0, max=(y.max() - y.min()) * 5
        )
        params.add(
            f"p{i}_center",
            expr=f"{theory_pos} * (1 - lattice_strain)",
            min=theory_pos * 0.97,
            max=theory_pos * 1.03,
        )
        params.add(f"p{i}_sigma", value=sigma, min=0.01, max=1.0)

        if "fraction" in peak_model.param_names:
            params.add(f"p{i}_fraction", value=0.5, min=0, max=1)

    print(f"理論ピーク数: {i + 1}")
    print(f"格子歪み補正: {'有効' if allow_lattice_strain else '無効'}")

    # フィッティング実行
    print("\n=== フィッティング開始 ===")
    result = composite_model.fit(y, params, x=x, nan_policy="omit", max_nfev=10000)

    print(f"フィッティング成功: {result.success}")
    if not result.success:
        print(f"警告: {result.message}")

    if allow_lattice_strain:
        fitted_strain = result.params["lattice_strain"].value
        fitted_strain_stderr = result.params["lattice_strain"].stderr or 0
        print(
            f"\nフィッティングされた格子歪み: {fitted_strain * 100:.3f} ± {fitted_strain_stderr * 100:.3f} %"
        )
        if abs(fitted_strain) > 0.005:
            print(
                f"  → 理論位置から約{abs(fitted_strain) * 100:.2f}%のシフトが検出されました"
            )
            if fitted_strain > 0:
                print("  → 引張応力または格子定数の減少を示唆")
            else:
                print("  → 圧縮応力または格子定数の増加を示唆")

    return result


def wppf_multi_phase(
    x,
    y,
    theoretical_peaks_list,
    phase_names,
    model_class,
    background_type="spline",
    num_knots=10,
    peak_matching_tolerance=0.5,
):
    """
    アプローチ3改: 複数相対応WPPF
    複数のCIFファイルから得られた理論ピークを使用

    Parameters
    ----------
    x : array_like
        実験データのx軸（2θ）
    y : array_like
        実験データの強度
    theoretical_peaks_list : list of polars.DataFrame
        各相の理論ピーク情報のリスト
    phase_names : list of str
        各相の名前のリスト
    model_class : class
        使用するピークモデル
    background_type : str
        バックグラウンドモデル
    num_knots : int
        B-splineノット数
    peak_matching_tolerance : float
        ピークマッチング許容範囲

    Returns
    -------
    result : ModelResult
        フィッティング結果（phase_assignments属性付き）
    """
    print("=== Multi-Phase WPPF ===")
    print(f"相の数: {len(theoretical_peaks_list)}")
    for i, name in enumerate(phase_names):
        print(f"  {name}: {len(theoretical_peaks_list[i])} 個の理論ピーク")

    # 実験ピーク検出
    min_peak_height = (y.max() - y.min()) * 0.05
    exp_peak_indices, _ = find_peaks(
        y, height=min_peak_height, prominence=(y.max() - y.min()) * 0.03, distance=10
    )

    exp_peak_positions = x[exp_peak_indices]
    print(f"\n実験データで検出されたピーク数: {len(exp_peak_positions)}")

    # 各実験ピークを相に割り当て
    phase_assignments = []

    for exp_idx, exp_pos in zip(exp_peak_indices, exp_peak_positions):
        best_match = None
        min_distance = float("inf")
        best_phase_idx = -1

        # すべての相の理論ピークと比較
        for phase_idx, (peaks_df, phase_name) in enumerate(
            zip(theoretical_peaks_list, phase_names)
        ):
            for row in peaks_df.iter_rows(named=True):
                theory_pos = row["two_theta"]
                distance = abs(exp_pos - theory_pos)
                if distance < min_distance and distance < peak_matching_tolerance:
                    min_distance = distance
                    best_match = row
                    best_phase_idx = phase_idx

        if best_match is not None:
            phase_assignments.append(
                {
                    "exp_position": exp_pos,
                    "exp_index": exp_idx,
                    "theory_position": best_match["two_theta"],
                    "phase": phase_names[best_phase_idx],
                    "phase_index": best_phase_idx,
                    "distance": min_distance,
                }
            )
        else:
            phase_assignments.append(
                {
                    "exp_position": exp_pos,
                    "exp_index": exp_idx,
                    "theory_position": None,
                    "phase": "Unassigned",
                    "phase_index": -1,
                    "distance": None,
                }
            )

    # 各相のピーク数を集計
    print("\n相割り当て結果:")
    for phase_name in phase_names + ["Unassigned"]:
        count = sum(1 for p in phase_assignments if p["phase"] == phase_name)
        print(f"  {phase_name}: {count} ピーク")

    # バックグラウンドモデル
    if background_type == "spline":
        knot_positions = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]
        background_model = SplineModel(prefix="bg_", xknots=knot_positions)
        params = background_model.guess(y, x=x)
    else:
        background_model = LinearModel(prefix="bg_")
        params = background_model.make_params()
        params["bg_slope"].set(value=0)
        params["bg_intercept"].set(value=y.min())

    composite_model = background_model

    # 各実験ピークでモデル構築
    for i, assignment in enumerate(phase_assignments):
        peak_model = model_class(prefix=f"p{i}_")
        composite_model += peak_model

        exp_pos = assignment["exp_position"]
        exp_idx = assignment["exp_index"]
        theory_pos = assignment.get("theory_position")

        amplitude_guess = y[exp_idx] - y.min()

        # FWHM推定
        half_height = y.min() + amplitude_guess / 2
        left_idx = exp_idx
        right_idx = exp_idx
        while left_idx > 0 and y[left_idx] > half_height:
            left_idx -= 1
        while right_idx < len(y) - 1 and y[right_idx] > half_height:
            right_idx += 1
        fwhm = abs(x[right_idx] - x[left_idx])
        if fwhm < 0.05:
            fwhm = 0.2
        sigma = fwhm / 2.355

        params.update(peak_model.make_params())

        # 理論位置がある場合は制約、ない場合は実験位置周辺で自由
        if theory_pos is not None:
            center_min = min(exp_pos, theory_pos) - 0.3
            center_max = max(exp_pos, theory_pos) + 0.3
        else:
            center_min = exp_pos - 0.2
            center_max = exp_pos + 0.2

        params[f"p{i}_amplitude"].set(
            value=amplitude_guess, min=0, max=amplitude_guess * 10
        )
        params[f"p{i}_center"].set(value=exp_pos, min=center_min, max=center_max)
        params[f"p{i}_sigma"].set(value=sigma, min=0.01, max=1.0)

        if "fraction" in peak_model.param_names:
            params[f"p{i}_fraction"].set(value=0.5, min=0, max=1)

    print(f"\n総ピークモデル数: {len(phase_assignments)}")

    # フィッティング実行
    print("\n=== フィッティング開始 ===")
    result = composite_model.fit(y, params, x=x, nan_policy="omit", max_nfev=10000)

    print(f"フィッティング成功: {result.success}")
    if not result.success:
        print(f"警告: {result.message}")

    result.phase_assignments = phase_assignments

    return result


def wppf_progressive(
    x,
    y,
    theoretical_peaks_df,
    model_class,
    background_type="spline",
    num_knots=10,
    intensity_threshold=10.0,
):
    """
    アプローチ4: 段階的WPPF
    まず強いピークでフィッティング、次に弱いピークを追加

    Parameters
    ----------
    x : array_like
        実験データのx軸（2θ）
    y : array_like
        実験データの強度
    theoretical_peaks_df : polars.DataFrame
        理論ピーク情報
    model_class : class
        使用するピークモデル
    background_type : str
        バックグラウンドモデル
    num_knots : int
        B-splineノット数
    intensity_threshold : float
        第1段階で使用する最小相対強度（%）

    Returns
    -------
    result : ModelResult
        フィッティング結果
    """
    print("=== Progressive WPPF ===")

    # ピークを強度で分類
    strong_peaks = theoretical_peaks_df.filter(
        pl.col("relative_intensity") >= intensity_threshold
    )
    weak_peaks = theoretical_peaks_df.filter(
        pl.col("relative_intensity") < intensity_threshold
    )

    print(f"強いピーク数: {len(strong_peaks)} (相対強度 >= {intensity_threshold}%)")
    print(f"弱いピーク数: {len(weak_peaks)}")

    if len(strong_peaks) == 0:
        print("警告: 強いピークが見つかりません。intensity_thresholdを下げてください")
        return None

    # 第1段階: 強いピークのみでハイブリッドフィッティング
    print("\n=== 第1段階: 強いピークのみ ===")
    result_stage1 = wppf_hybrid_thin_film(
        x,
        y,
        strong_peaks,
        model_class,
        background_type=background_type,
        num_knots=num_knots,
        peak_matching_tolerance=0.5,
    )

    if result_stage1 is None or not result_stage1.success:
        print("第1段階のフィッティングが失敗しました")
        return result_stage1

    print(f"\n第1段階完了: R² = {result_stage1.rsquared:.4f}")

    # 第2段階は必要に応じて実装（現状は第1段階のみ）
    return result_stage1


def plot_wppf_result_extended(x, y, result, theoretical_peaks_df=None, method="hybrid"):
    """
    拡張版WPPFフィッティング結果をプロット
    マッチング情報や相情報を表示

    Parameters
    ----------
    x : array_like
        実験データのx軸
    y : array_like
        実験データの強度
    result : ModelResult
        フィッティング結果
    theoretical_peaks_df : polars.DataFrame, optional
        理論ピーク情報
    method : str
        フィッティング方法名
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上段: データとフィッティング結果
    ax1.plot(x, y, "b.", label="Experimental Data", markersize=3, alpha=0.5)
    ax1.plot(x, result.best_fit, "r-", label="Fit", linewidth=2)

    # コンポーネント
    comps = result.eval_components(x=x)

    # ハイブリッド法の場合、マッチング情報を使用
    if hasattr(result, "matched_peaks"):
        matched_peaks = result.matched_peaks
        unmatched_peaks = result.unmatched_peaks

        # マッチしたピーク
        for i, peak_info in enumerate(matched_peaks):
            if f"p{i}_" in comps:
                exp_pos = peak_info["exp_position"]
                theory_pos = peak_info["theory_position"]
                fitted_center = result.params[f"p{i}_center"].value
                ax1.plot(
                    x,
                    comps[f"p{i}_"],
                    "--",
                    linewidth=1,
                    alpha=0.7,
                    label=f"P{i + 1} (exp:{exp_pos:.2f}°, theory:{theory_pos:.2f}°)",
                )

        # マッチしなかったピーク
        for i, peak_info in enumerate(unmatched_peaks):
            j = len(matched_peaks) + i
            if f"p{j}_" in comps:
                exp_pos = peak_info["exp_position"]
                ax1.plot(
                    x,
                    comps[f"p{j}_"],
                    ":",
                    linewidth=1,
                    alpha=0.7,
                    label=f"P{j + 1} (unmatched, {exp_pos:.2f}°)",
                )

    # 複数相の場合
    elif hasattr(result, "phase_assignments"):
        phase_assignments = result.phase_assignments
        phase_colors = plt.cm.tab10(np.arange(10))

        for i, assignment in enumerate(phase_assignments):
            if f"p{i}_" in comps:
                phase = assignment["phase"]
                exp_pos = assignment["exp_position"]
                # 相ごとに色を変える
                phase_idx = assignment.get("phase_index", -1)
                color = phase_colors[phase_idx % 10] if phase_idx >= 0 else "gray"
                ax1.plot(
                    x,
                    comps[f"p{i}_"],
                    "--",
                    color=color,
                    linewidth=1,
                    alpha=0.7,
                    label=f"P{i + 1} ({phase}, {exp_pos:.2f}°)",
                )

    # 通常の場合
    elif theoretical_peaks_df is not None:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(theoretical_peaks_df)))
        for i, (row, color) in enumerate(
            zip(theoretical_peaks_df.iter_rows(named=True), colors)
        ):
            if f"p{i}_" in comps:
                theory_pos = row["two_theta"]
                fitted_center = result.params[f"p{i}_center"].value
                ax1.plot(
                    x,
                    comps[f"p{i}_"],
                    "--",
                    color=color,
                    linewidth=1,
                    alpha=0.7,
                    label=f"P{i + 1} (theory:{theory_pos:.2f}°, fit:{fitted_center:.2f}°)",
                )

    # バックグラウンド
    if "bg_" in comps:
        ax1.plot(x, comps["bg_"], "k:", label="Background", linewidth=2)

    ax1.set_xlabel("2θ (degrees)", fontsize=12)
    ax1.set_ylabel("Intensity (cps)", fontsize=12)
    ax1.set_title(f"WPPF Result - {method}", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)

    # 下段: 残差
    residual = y - result.best_fit
    ax2.plot(x, residual, "g-", linewidth=1)
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=1)
    ax2.fill_between(x, 0, residual, alpha=0.3, color="green")
    ax2.set_xlabel("2θ (degrees)", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.set_title("Fit Residuals", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 統計情報
    print("\n=== Fitting Statistics ===")
    print(f"Chi-squared: {result.chisqr:.4e}")
    print(f"Reduced chi-squared: {result.redchi:.4e}")
    print(f"R-squared (R²): {result.rsquared:.6f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")

    # R-factors
    y_obs = y
    y_calc = result.best_fit
    rp = np.sum(np.abs(y_obs - y_calc)) / np.sum(y_obs) * 100
    weights = 1 / np.where(y_obs > 0, y_obs, 1)
    rwp = (
        np.sqrt(np.sum(weights * (y_obs - y_calc) ** 2) / np.sum(weights * y_obs**2))
        * 100
    )

    print("\nR-factors:")
    print(f"  Rp:  {rp:.2f}%")
    print(f"  Rwp: {rwp:.2f}%")

    # 格子歪み情報
    if "lattice_strain" in result.params:
        strain = result.params["lattice_strain"].value
        strain_stderr = result.params["lattice_strain"].stderr or 0
        print(f"\n格子歪み: {strain * 100:.3f} ± {strain_stderr * 100:.3f} %")

    return result
