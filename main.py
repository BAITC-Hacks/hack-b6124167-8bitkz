#!/usr/bin/env python3
"""Точка входа для запуска приложения.

Использование:
    python main.py

Приложение запустится на http://localhost:8501
"""
import logging
import subprocess
import sys
from pathlib import Path

# Подавляем warnings asyncio о разрыве соединения (WinError 10054)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def main():
    """Запускает Streamlit приложение."""
    app_path = Path(__file__).parent / "app" / "main.py"

    if not app_path.exists():
        print(f"Ошибка: файл приложения не найден: {app_path}")
        sys.exit(1)

    print("Запуск GovTech Camp — мониторинг стационаров...")
    print("Приложение откроется в браузере: http://localhost:8501")
    print("Для остановки нажмите Ctrl+C\n")

    try:
        # Запускаем streamlit
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=Path(__file__).parent
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
        sys.exit(0)


if __name__ == "__main__":
    main()
