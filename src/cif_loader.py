from pathlib import Path

import numpy as np
import polars as pl
import xrayutilities as xu


def get_theoretical_peaks_from_cif(
    cif_path, wavelength="CuKa1", two_theta_range=(10, 90), min_intensity=0.01
):
    """
    CIFファイルから理論的なXRDピーク位置と相対強度を取得

    Parameters
    ----------
    cif_path : str or Path
        CIFファイルのパス
    wavelength : str or float, optional
        X線波長。'CuKa1', 'CuKa2', 'CuKa'（平均）または波長値（Å）
        デフォルト: 'CuKa1' (1.5406 Å)
    two_theta_range : tuple, optional
        計算する2θ範囲 (min, max)
    min_intensity : float, optional
        表示する最小相対強度（最大強度に対する比率）

    Returns
    -------
    peaks_df : polars.DataFrame
        ピーク情報（2theta, intensity, hkl, d-spacing）
    """
    # CIFファイルを読み込む
    cif_path = Path(cif_path)
    if not cif_path.exists():
        raise FileNotFoundError(f"CIFファイルが見つかりません: {cif_path}")

    print(f"CIFファイルを読み込み中: {cif_path.name}")

    # xrayutilitiesでCIFを読み込み
    crystal = xu.materials.Crystal.fromCIF(str(cif_path))
    print(f"結晶系: {crystal.lattice.space_group}")
    print(f"格子定数: a={crystal.lattice.a:.4f} Å")

    # 波長を設定
    if wavelength == "CuKa1":
        wl = xu.wavelength("CuKa1")  # 1.5406 Å
        en = xu.utilities.energy("CuKa1")  # energy in eV
    elif wavelength == "CuKa2":
        wl = xu.wavelength("CuKa2")  # 1.5444 Å
        en = xu.utilities.energy("CuKa2")
    elif wavelength == "CuKa":
        wl = xu.wavelength("CuKa")  # 加重平均
        en = xu.utilities.energy("CuKa")
    else:
        wl = float(wavelength)
        en = xu.utilities.energy(wl)

    print(f"波長: {wl:.4f} Å ({wavelength})")

    # 2θ範囲を設定
    two_theta_min, two_theta_max = two_theta_range

    # Powderクラスで材料を定義し、PowderModelで計算
    # Powderクラスは粉末試料の体積、結晶子サイズ、ひずみなどを設定
    powder = xu.simpack.Powder(crystal, 1)  # volume=1 (相対値)

    # PowderModelで理論パターンを計算
    pm = xu.simpack.PowderModel(powder, I0=100, en=en)

    # 2θ範囲のデータポイントを生成
    tt_array = np.linspace(two_theta_min, two_theta_max, 8000)

    print("理論XRDパターンを計算中...")

    # 粉末回折パターンを計算
    try:
        intensity_profile = pm.simulate(tt_array)
    finally:
        pm.close()  # マルチプロセス処理のクリーンアップ

    # ピーク位置を検出（scipy.signal.find_peaksを使用）
    from scipy.signal import find_peaks

    # 最小強度の閾値を設定
    threshold = (
        min_intensity * np.max(intensity_profile)
        if np.max(intensity_profile) > 0
        else 0
    )

    peaks_indices, peak_properties = find_peaks(
        intensity_profile, height=threshold, prominence=0.1
    )

    if len(peaks_indices) == 0:
        print("ピークが見つかりませんでした")
        return None

    print(f"検出されたピーク数: {len(peaks_indices)}")

    # ピーク情報を抽出
    peak_data = []
    for idx in peaks_indices:
        two_theta = tt_array[idx]
        intensity = intensity_profile[idx]

        # d-spacingを計算
        theta = np.radians(two_theta / 2)
        d_spacing = wl / (2 * np.sin(theta))

        peak_data.append(
            {
                "two_theta": two_theta,
                "intensity": intensity,
                "d_spacing": d_spacing,
            }
        )

    # DataFrameを作成
    peaks_df = pl.DataFrame(peak_data)

    # 強度を正規化（最大値を100に）
    max_intensity = peaks_df["intensity"].max()
    peaks_df = peaks_df.with_columns(
        (pl.col("intensity") / max_intensity * 100).alias("relative_intensity")
    )

    # 2θでソート
    peaks_df = peaks_df.sort("two_theta")

    print(f"検出されたピーク数: {len(peaks_df)}")
    print(
        f"2θ範囲: {peaks_df['two_theta'].min():.2f} - {peaks_df['two_theta'].max():.2f}°"
    )

    return peaks_df


if __name__ == "__main__":
    # 使用例
    cif_path = Path("data/cif/1538066.cif")  # NbTi
    peaks_df = get_theoretical_peaks_from_cif(
        cif_path, wavelength="CuKa1", two_theta_range=(10, 90), min_intensity=0.01
    )

    if peaks_df is not None:
        print("\n=== 理論ピーク（相対強度 > 1%） ===")
        print(
            peaks_df.filter(pl.col("relative_intensity") > 1.0).select(
                ["two_theta", "relative_intensity", "d_spacing"]
            )
        )
