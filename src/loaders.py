"""Загрузка и первичная нормализация сырых данных кейса 1.

Схема работы: сырые CSV читаются один раз и сохраняются в Parquet-кэш
(cache/). Все последующие загрузки идут из бинарного кэша — в ~10 раз
быстрее разбора текста. Кэш автоматически перестраивается, если CSV
новее кэша или его не хватает.
"""
import glob
import os.path

import pandas as pd

from . import config
from .regions import add_region_column, parse_region

TEXT_COLS = {
    "referrals": ["hospitalization_code", "referring_mo", "hospital_mo", "icd10_ref_diag_code",
                  "diagnosis_name", "bed_profile", "territorial_type", "referral_purpose",
                  "finance_source"],
    "queue": ["region_origin_code", "mo_destination_code", "profile_code", "icd10_ref_diag_code",
              "diagnosis_name", "operation_code", "operation_name"],
    "refusals": ["region_in", "org_in", "resident", "insured", "benefit_cat", "attach_region",
                 "attach_org", "icd10", "icd_name", "finance_src"],
    "treated": ["medicine_organization"],
}


def _files(pattern: str) -> list[str]:
    files = sorted(glob.glob(str(config.RAW_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"Нет файлов по шаблону {pattern} в {config.RAW_DIR}")
    return files


def _cache_fresh(dataset: str, csv_files: list[str]) -> bool:
    cache = config.CACHE_DIR / f"{dataset}.parquet"
    if not cache.exists():
        return False
    newest_csv = max(os.path.getmtime(f) for f in csv_files)
    return cache.stat().st_mtime >= newest_csv


def _read(pattern: str, dataset: str) -> pd.DataFrame:
    config.CACHE_DIR.mkdir(exist_ok=True)
    csv_files = _files(pattern)

    if _cache_fresh(dataset, csv_files):
        return pd.read_parquet(config.CACHE_DIR / f"{dataset}.parquet")

    frames = []
    for path in csv_files:
        frames.append(pd.read_csv(path, encoding="utf-8-sig",
                                  dtype={c: "category" for c in TEXT_COLS[dataset]}))
    df = pd.concat(frames, ignore_index=True)
    for col in config.DT_COLS.get(dataset, []):
        # ISO8601 единым парсером — заметно быстрее вывода формата по каждой строке
        df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")

    df.to_parquet(config.CACHE_DIR / f"{dataset}.parquet", index=False)
    return df


def load_referrals() -> pd.DataFrame:
    """Направления на плановую госпитализацию (ИС БГ), ~767 тыс. строк, 2025 Q1.

    Добавляет целевые признаки:
      outcome      — hospitalized / refused / pending (нет ни даты госпитализации, ни отказа)
      waiting_days — дней от регистрации до госпитализации (для hospitalized)
      offer_days   — дней от регистрации до плановой даты
    """
    df = _read(config.F_REFERRALS, "referrals")
    df["outcome"] = "pending"
    df.loc[df["refusal_dt"].notna(), "outcome"] = "refused"
    df.loc[df["refusal_dt"].isna() & df["hospitalization_dt"].notna(), "outcome"] = "hospitalized"
    df["outcome"] = df["outcome"].astype("category")
    df["waiting_days"] = (df["hospitalization_dt"] - df["registration_dt"]).dt.total_seconds() / 86400
    df["offer_days"] = (df["planned_dt"] - df["registration_dt"]).dt.total_seconds() / 86400
    df["icd_group"] = df["icd10_ref_diag_code"].astype(str).str.slice(0, 1)
    df = add_region_column(df, "hospital_mo", "region")
    return df


def load_queue() -> pd.DataFrame:
    """Ожидающие плановую госпитализацию (ИС БГ), ~765 тыс. строк — снимок очереди."""
    df = _read(config.F_QUEUE, "queue")
    # Позиция в очереди: бывают значения с суффиксом вида "9S" (повторная постановка)
    df["patient_seq_no"] = pd.to_numeric(df["patient_seq_no"], errors="coerce")
    return df


def load_refusals() -> pd.DataFrame:
    """Отказы в плановой госпитализации, приёмный покой (ИС БГ), ~1.5 млн строк."""
    df = _read(config.F_REFUSALS, "refusals")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def load_treated() -> pd.DataFrame:
    """Пролеченные случаи в разрезе МО (ЭРСБ) — 2 196 агрегатов, снимок без периода."""
    df = _read(config.F_TREATED, "treated")
    num = ["discharged_total", "discharged_children", "treated_budget", "treated_paid",
           "discharged_within_day", "deaths_total", "bed_days"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["amount_to_pay"] = pd.to_numeric(df["amount_to_pay"], errors="coerce")
    df["bed_days_per_case"] = df["bed_days"] / df["discharged_total"].replace(0, pd.NA)
    df["deaths_share"] = df["deaths_total"] / df["discharged_total"].replace(0, pd.NA)
    df = add_region_column(df, "medicine_organization", "region")
    return df


LOADERS = {
    "referrals": load_referrals,
    "queue": load_queue,
    "refusals": load_refusals,
    "treated": load_treated,
}
