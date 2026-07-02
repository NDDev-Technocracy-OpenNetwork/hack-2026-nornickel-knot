"""Copies a small, hand-picked cross-section of the corpus into data/sample/
for fast MVP iteration, without touching the full 1300+ document archive."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "yandex" / "Задача 2. Научный клубок" / "Источники информации"
DEST_ROOT = ROOT / "data" / "sample"

SAMPLE_FILES = [
    "Доклады/Тяпкина ПА_Пермь_Зимняя школа.pdf",
    "Доклады/Доклад_Вязовой О.Н.pdf",
    "Доклады/Доклад_Вострикова Н.М.pdf",
    "Доклады/Доклад_Луканина М.А.pdf",
    "Доклады/Доклад_Новожилова О.С.pdf",
    "Журналы/Цветные металлы/2016/demin1.pdf",
    "Журналы/Обогащение руд/2018/ОР 01_18.pdf",
    "Журналы/Обогащение руд/2019/ОР 03_19.pdf",
    "Журналы/Обогащение руд/2019/ОР 04_19.pdf",
    "Журналы/Обогащение руд/2024/ОР № 5-24.pdf",
    "Материалы конференций/Источники данных о меди/CRU/CRU-2011/CopperMonitor2011Sep.pdf",
    "Материалы конференций/Источники данных о меди/Antaike-China/Antaike-2011/Antaike-2011-Cu report.pdf",
    "Материалы конференций/Источники данных о МПГ/Ежемесячные издания/Mitsui/2011/август/MitsuiMonthlyAug2011.pdf",
    "Материалы конференций/Источники данных о МПГ/Квартальные издания/CRU/2007/Quaterly/GoldQ12007.pdf",
    "Обзоры/Sheba's Ridge.pdf",
    "Обзоры/BCL 2011.pdf",
    "Обзоры/Bindura_2010.pdf",
    "Обзоры/Cu 2011.pdf",
]


def main() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel_path in SAMPLE_FILES:
        src = SOURCE_ROOT / rel_path
        category = rel_path.split("/", 1)[0]
        dest_dir = DEST_ROOT / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(rel_path).name
        if not src.exists():
            print(f"MISSING: {src}")
            continue
        shutil.copy2(src, dest)
        copied += 1
        print(f"OK: {dest.relative_to(ROOT)}")
    print(f"\nCopied {copied}/{len(SAMPLE_FILES)} files into {DEST_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
