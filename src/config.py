"""Конфигурация проекта: пути и параметры данных кейса 1 GovTech Camp."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "датасет"
CACHE_DIR = PROJECT_ROOT / "cache"

# Имена исходных файлов (портал ashyq.data.gov.kz, выгрузка 2025 Q1)
F_REFERRALS = "Направления на плановую госпитализацию в стационары_part_*_of_003.csv"
F_QUEUE = "Ожидающие плановую госпитализацию в стационары.csv"
F_REFUSALS = "Отказы в плановой госпитализации (приёмный покой)_part_*_of_006.csv"
F_TREATED = "Количество пролеченных случаев в разрезе МО.csv"

DT_COLS = {
    "referrals": ["registration_dt", "planned_dt", "polyclinic_dt", "hospitalization_dt", "refusal_dt"],
    "queue": ["registration_dt", "planned_dt"],
    "refusals": ["refuse_dt"],
    "treated": [],
}
