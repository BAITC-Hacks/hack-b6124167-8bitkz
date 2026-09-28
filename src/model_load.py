"""Модель №2: прогноз ежедневных госпитализаций (нагрузка стационаров).

Считаем сколько пациентов госпитализируется по паре (стационар, профиль коек)
в каждый день; LightGBM на лаговых признаках. Валидация по времени: обучение
до 18 марта, проверка — последние 14 дней марта. Прогноз на 14 дней вперёд —
рекурсивный. Артефакты: models/load_model.txt + load_meta.json.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, loaders

MODEL_DIR = (config.PROJECT_ROOT / "models").resolve()
BOOSTER_PATH = MODEL_DIR / "load_model.txt"
META_PATH = MODEL_DIR / "load_meta.json"

HORIZON = 14
CUTOFF = pd.Timestamp("2025-03-18")
HIST_START = pd.Timestamp("2025-01-01")
HIST_END = pd.Timestamp("2025-04-01")  # выгрузка регистраций обрезана 31.03.2025
LAGS = [1, 7, 14]
FEATURES = ["mo", "profile", "dow", "dom", "lag1", "lag7", "lag14", "roll7", "series_mean"]
CAT_FEATURES = ["mo", "profile"]


def build_counts(ref: pd.DataFrame) -> pd.DataFrame:
    """Ежедневные госпитализации по парам (стационар, профиль коек) в окне Q1 2025.

    hospitalization_dt в выгрузке размазан по 2024–2026 (записи, обновлённые
    позже окна экспорта) — берём только госпитализации января–марта 2025.
    """
    ev = ref[ref["hospitalization_dt"].notna()].copy()
    ev = ev[ev["hospitalization_dt"].between(HIST_START, HIST_END)]
    ev["date"] = ev["hospitalization_dt"].dt.normalize()
    counts = (ev.groupby(["date", "hospital_mo", "bed_profile"], observed=True)
                .size().reset_index(name="count"))
    counts.columns = ["date", "mo", "profile", "count"]
    return counts


def _add_features(df: pd.DataFrame, group_mean: dict) -> pd.DataFrame:
    df = df.sort_values(["mo", "profile", "date"])
    g = df.groupby(["mo", "profile"], observed=True)["count"]
    for lag in LAGS:
        df[f"lag{lag}"] = g.shift(lag)
    df["roll7"] = g.shift(1).rolling(7, min_periods=3).mean()
    df["dow"] = df["date"].dt.dayofweek.astype(int)
    df["dom"] = df["date"].dt.day.astype(int)
    key = df["mo"].astype(str) + "||" + df["profile"].astype(str)
    df["series_mean"] = key.map(group_mean).astype(float)
    return df


def _codes(series: pd.DataFrame, cats: dict, f: str) -> np.ndarray:
    return pd.Categorical(series[f].astype(str), categories=cats[f]).codes.astype("int32")


def _matrix(meta: dict, df: pd.DataFrame) -> np.ndarray:
    cols = []
    for f in meta["features"]:
        if f in meta["cat_features"]:
            cols.append(_codes(df, meta["categories"], f))
        else:
            cols.append(df[f].astype("float32").values)
    return np.column_stack(cols)


def train(ref: pd.DataFrame | None = None, save: bool = True) -> dict:
    ref = loaders.load_referrals() if ref is None else ref
    t0 = time.time()
    counts = build_counts(ref)

    # среднее по серии только по обучающему периоду — без утечки будущего
    train_counts = counts[counts["date"] < CUTOFF]
    group_mean = (train_counts.assign(k=lambda d: d["mo"].astype(str) + "||" + d["profile"].astype(str))
                  .groupby("k")["count"].mean().round(3).to_dict())

    df = _add_features(counts, group_mean).dropna(subset=["lag1", "roll7"])
    for c in CAT_FEATURES:
        cats_sorted = sorted(df[c].dropna().astype(str).unique().tolist())
        df[c] = pd.Categorical(df[c].astype(str), categories=cats_sorted)

    train, test = df[df["date"] < CUTOFF], df[df["date"] >= CUTOFF]
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=20, n_jobs=-1, verbose=-1, random_state=42,
    )
    model.fit(train[FEATURES], train["count"], categorical_feature=CAT_FEATURES)
    pred = np.maximum(model.predict(test[FEATURES]), 0)
    y = test["count"].values
    wape = float(np.abs(pred - y).sum() / max(y.sum(), 1))
    mae = float(np.abs(pred - y).mean())

    cats = {c: list(df[c].cat.categories) for c in CAT_FEATURES}
    resid_std = float(np.std(pred - y))
    meta = {
        "features": FEATURES, "cat_features": CAT_FEATURES, "categories": cats,
        "group_mean": group_mean, "resid_std": round(resid_std, 3),
        "metrics": {"wape": round(wape, 4), "mae": round(mae, 3),
                    "n_train": len(train), "n_test": len(test)},
        "cutoff": CUTOFF.strftime("%d.%m.%Y"),
        "last_date": counts["date"].max().strftime("%Y-%m-%d"),
        "horizon": HORIZON,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_seconds": round(time.time() - t0, 1),
        "importance": [[k, float(v)] for k, v in
                       pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
                       .sort_values(ascending=False).items()],
    }
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        model.booster_.save_model(str(BOOSTER_PATH))
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    booster = lgb.Booster(model_file=str(BOOSTER_PATH)) if save else model.booster_
    return {**meta, "booster": booster}


def load_or_train() -> dict:
    if BOOSTER_PATH.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        meta["booster"] = lgb.Booster(model_file=str(BOOSTER_PATH))
        return meta
    return train()


def forecast(artifact: dict, counts: pd.DataFrame, mo: str, profile: str,
             horizon: int | None = None) -> pd.DataFrame:
    """Рекурсивный прогноз на horizon дней для пары (стационар, профиль)."""
    horizon = horizon or artifact["horizon"]
    hist = counts[(counts["mo"] == mo) & (counts["profile"] == profile)].sort_values("date")
    if hist.empty:
        return pd.DataFrame(columns=["date", "yhat", "lo", "hi"])
    known = {r.date().isoformat(): float(c) for r, c in zip(hist["date"], hist["count"])}
    series_mean = artifact["group_mean"].get(f"{mo}||{profile}", 0.0)
    last = hist["date"].max()
    std = artifact["resid_std"]

    future, dates = [], []
    for h in range(1, horizon + 1):
        d = last + pd.Timedelta(days=h)
        lag = lambda n: known.get((d - pd.Timedelta(days=n)).date().isoformat(),
                                  series_mean if h > n else 0.0)
        roll7 = np.mean([known.get((d - pd.Timedelta(days=k)).date().isoformat(), series_mean)
                         for k in range(1, 8)])
        X = pd.DataFrame([{ "mo": mo, "profile": profile,
                            "dow": d.dayofweek, "dom": d.day,
                            "lag1": lag(1), "lag7": lag(7), "lag14": lag(14),
                            "roll7": roll7, "series_mean": series_mean }])[artifact["features"]]
        y = max(float(artifact["booster"].predict(_matrix(artifact, X))[0]), 0.0)
        known[d.date().isoformat()] = y
        future.append(y)
        dates.append(d)

    return pd.DataFrame({
        "date": dates, "yhat": future,
        "lo": np.maximum(np.array(future) - 1.28 * std, 0),
        "hi": np.array(future) + 1.28 * std,
    })
