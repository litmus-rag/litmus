"""Step 13: assemble all tables into one summary artifact, ready to paste
into the paper (eval_plan.md §5/§6 Step 13).
"""

from __future__ import annotations

import json

from eval_study import config


def main() -> None:
    tables = {}
    for name in ["table1", "table2", "table3", "table4", "table5", "table6"]:
        path = config.TABLES_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} -- re-run the corresponding study step first")
        tables[name] = json.load(open(path))

    out_path = config.TABLES_DIR / "all_tables_summary.json"
    with open(out_path, "w") as f:
        json.dump(tables, f, indent=2)
    print(f"Wrote {out_path}")

    for name, data in tables.items():
        assert data, f"{name} is empty"
    print(f"All {len(tables)} tables present and non-empty.")


if __name__ == "__main__":
    main()
