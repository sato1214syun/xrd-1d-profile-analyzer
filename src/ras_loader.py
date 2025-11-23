from pathlib import Path
import polars as pl
import xrayutilities as xu


def load_ras_file(file_path):
    """
    RASファイルを読み込み、Polars DataFrameとして返す。

    Parameters
    ----------
    file_path : str or Path
        RASファイルのパス

    Returns
    -------
    pl.DataFrame
        x (2theta), counts, cps を含むDataFrame
    """
    ras_path = Path(file_path)
    if not ras_path.exists():
        raise FileNotFoundError(f"File not found: {ras_path}")

    ras_data = xu.io.RASFile(str(ras_path))
    step_width = ras_data.scan.meas_step
    # meas_speed is usually in deg/min. Converting to deg/sec if needed,
    # or just following original logic: scan_speed = ras_data.scan.meas_speed / 60
    scan_speed = ras_data.scan.meas_speed / 60

    ras_df = (
        pl.DataFrame(ras_data.scan.data, schema=["x", "y", "attenuator"])
        .select(
            pl.col("x"),
            (pl.col("y") * pl.col("attenuator")).alias("counts"),
        )
        .with_columns(
            (pl.col("counts") / step_width * scan_speed).alias("cps"),
        )
    )
    return ras_df
