"""Rebuild the SQLite FTS5 index from the CSV dataset.

Usage: python -m scripts.build_index
"""
from app.config import get_settings
from app.dataset import build_index


def main() -> None:
    s = get_settings()
    n = build_index(s.dataset_csv, s.db_path)
    print(f"Indexed {n} documents from {s.dataset_csv} -> {s.db_path}")


if __name__ == "__main__":
    main()
