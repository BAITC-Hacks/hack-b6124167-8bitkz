"""Детектор аномальных всплесков отказов в госпитализации.

Базовая линия — скользящая медиана предыдущих 14 дней по региону, разброс —
MAD (медианное абсолютное отклонение). Всплеск: z >= 3.5 и выше базовой линии.
MAD выбран вместо стандартного отклонения, чтобы редкие мощные выбросы не
раздували порог для самих себя.
"""
import numpy as np
import pandas as pd

WINDOW = 14
Z_THRESHOLD = 3.5


def daily_by_region(refusals: pd.DataFrame) -> pd.DataFrame:
    ev = refusals[refusals["refuse_dt"].notna()]
    return (ev.assign(date=ev["refuse_dt"].dt.normalize())
              .groupby(["region_in", "date"], observed=True).size()
              .reset_index(name="count")
              .sort_values(["region_in", "date"]))


def detect_spikes(daily: pd.DataFrame) -> pd.DataFrame:
    """К дням каждого региона добавляет baseline, z и флаг всплеска."""
    daily = daily.sort_values(["region_in", "date"]).copy()
    g = daily.groupby("region_in", observed=True)["count"]
    past = g.shift(1)  # окно без текущего дня — иначе выброс сам себя маскирует
    baseline = past.rolling(WINDOW, min_periods=7).median().reset_index(level=0, drop=True)
    mad = (past - baseline).abs().rolling(WINDOW, min_periods=5).median().reset_index(level=0, drop=True)
    daily["baseline"] = baseline
    daily["z"] = (daily["count"] - baseline) / (1.4826 * mad + 1e-9)
    daily["spike"] = (daily["z"] >= Z_THRESHOLD) & (daily["count"] > daily["baseline"])
    return daily


def spikes_table(daily: pd.DataFrame, last_days: int = 30) -> pd.DataFrame:
    """Последние всплески по всем регионам, свежие сверху."""
    spikes = daily[daily["spike"]].copy()
    if spikes.empty:
        return pd.DataFrame(columns=["Регион", "Дата", "Отказов", "Базовый уровень", "z"])
    cutoff = daily["date"].max() - pd.Timedelta(days=last_days)
    spikes = spikes[spikes["date"] >= cutoff].sort_values(["date", "z"], ascending=[False, False])
    return pd.DataFrame({
        "Регион": spikes["region_in"].astype(str),
        "Дата": spikes["date"].dt.strftime("%d.%m.%Y"),
        "Отказов": spikes["count"].astype(int),
        "Базовый уровень": spikes["baseline"].round(0).astype(int),
        "z": spikes["z"].round(1),
    }).reset_index(drop=True)
