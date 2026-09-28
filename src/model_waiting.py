"""Модель №1: прогноз времени ожидания плановой госпитализации (в днях).

LightGBM-регрессия по признакам направления. Валидация по времени:
обучение на январе–феврале, проверка на марте 2025 (утечка исключена).
Артефакты: нативный формат LightGBM + JSON-метаданные (без pickle).
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, loaders

MODEL_DIR = (config.PROJECT_ROOT / "models").resolve()
BOOSTER_PATH = MODEL_DIR / "waiting_model.txt"
META_PATH = MODEL_DIR / "waiting_meta.json"

FEATURES = ["hospital_mo", "bed_profile", "icd_group", "region", "territorial_type",
            "referral_purpose", "finance_source", "month", "dow"]
CAT_FEATURES = FEATURES[:7]
TARGET = "waiting_days"
CUTOFF = pd.Timestamp("2025-03-01")  # train: до этой даты, test: после


def build_dataset(ref: pd.DataFrame) -> pd.DataFrame:
    """Госпитализированные с корректными сроками (0–90 дней), детерминированные категории."""
    df = ref[(ref["outcome"] == "hospitalized") & ref["waiting_days"].between(0, 90)].copy()
    df["month"] = df["registration_dt"].dt.month.astype(int)
    df["dow"] = df["registration_dt"].dt.dayofweek.astype(int)
    for c in CAT_FEATURES:
        cats = sorted(df[c].dropna().astype(str).unique().tolist())
        df[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return df


def _metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": round(float(np.mean(np.abs(err))), 3),
        "wape": round(float(np.sum(np.abs(err)) / np.sum(np.abs(y_true))), 4),
        "median_ae": round(float(np.median(np.abs(err))), 3),
        "within_1d": round(float(np.mean(np.abs(err) <= 1)), 4),
    }


def _to_matrix(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Категории -> коды по сохранённому при обучении списку категорий."""
    cols = {}
    for f in artifact["features"]:
        if f in artifact["cat_features"]:
            cats = artifact["categories"][f]
            idx = pd.Categorical(df[f].astype(str), categories=cats).codes
            cols[f] = idx.astype("int32")
        else:
            cols[f] = df[f].astype("int32").values
    return pd.DataFrame(cols, columns=artifact["features"])


def train(ref: pd.DataFrame | None = None, save: bool = True) -> dict:
    ref = loaders.load_referrals() if ref is None else ref
    t0 = time.time()
    df = build_dataset(ref)
    train, test = df[df["registration_dt"] < CUTOFF], df[df["registration_dt"] >= CUTOFF]

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=100, n_jobs=-1, verbose=-1, random_state=42,
    )
    model.fit(train[FEATURES], train[TARGET], categorical_feature=CAT_FEATURES)
    pred = model.predict(test[FEATURES])
    metrics = _metrics(test[TARGET], pred)
    metrics["n_train"], metrics["n_test"] = len(train), len(test)

    importance = (pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
                  .sort_values(ascending=False))

    artifact = {
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "categories": {c: list(df[c].cat.categories) for c in CAT_FEATURES},
        "metrics": metrics,
        "importance": [[k, float(v)] for k, v in importance.items()],
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_seconds": round(time.time() - t0, 1),
        "cutoff": CUTOFF.strftime("%d.%m.%Y"),
    }
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        model.booster_.save_model(str(BOOSTER_PATH))
        META_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    booster = lgb.Booster(model_file=str(BOOSTER_PATH)) if save else model.booster_
    return {**artifact, "booster": booster}


def load_or_train() -> dict:
    if BOOSTER_PATH.exists() and META_PATH.exists():
        artifact = json.loads(META_PATH.read_text(encoding="utf-8"))
        artifact["booster"] = lgb.Booster(model_file=str(BOOSTER_PATH))
        return artifact
    return train()


def make_row(artifact: dict, **values) -> tuple[pd.DataFrame, dict]:
    """Матрица признаков одного направления (категории -> коды обучения) + читаемые значения."""
    display = {f: str(values.get(f, "")) for f in artifact["features"]}
    return _to_matrix(artifact, pd.DataFrame([values])), display


def predict_one(artifact: dict, X: pd.DataFrame, display: dict | None = None,
                explainer=None) -> tuple[float, pd.DataFrame]:
    """Прогноз для одного направления + топ факторов через SHAP.

    display — исходные (читаемые) значения признаков; иначе покажутся коды.
    explainer можно передать извне (кэш), иначе создаётся на вызов.
    """
    import shap  # ленивый импорт: нужен только для объяснений

    booster: lgb.Booster = artifact["booster"]
    arr = X.to_numpy(dtype=np.float32)
    pred = float(booster.predict(arr)[0])
    if explainer is None:
        explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(arr)[0]
    contrib = (pd.DataFrame({
            "Признак": artifact["features"],
            "Вклад (дней)": sv,
            "Значение": [display[f] if display else str(v) for f, v in zip(artifact["features"], X.iloc[0])],
        })
        .loc[np.abs(sv).argsort()[::-1]]
        .reset_index(drop=True))
    return max(pred, 0.0), contrib
