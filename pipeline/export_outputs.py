# pipeline/export_outputs.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipeline.functions import HW2Config, build_cell_artifacts


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _materialize_artifacts(artifacts: dict[str, bytes], out_dir: Path) -> None:
    for rel, data in artifacts.items():
        fp = out_dir / rel
        _mkdir(fp.parent)
        fp.write_bytes(data)


def run_all(
    x_path: str,
    y_path: str,
    sheet_x: str,
    sheet_y: str,
    date_col: str,
    target_col: str,
    out_dir: str,
) -> Dict[str, Any]:
    """
    Integration entrypoint used by tests.
    Writes artifacts to out_dir and returns metrics dict.
    """
    out = Path(out_dir)
    _mkdir(out)

    df_x = pd.read_excel(x_path, sheet_name=sheet_x)
    df_y = pd.read_excel(y_path, sheet_name=sheet_y)

    cfg = HW2Config(date_col=date_col, target_col=target_col)

    artifacts = build_cell_artifacts(df_y=df_y, df_x=df_x, cfg=cfg)
    _materialize_artifacts(artifacts, out)

    metrics_path = out / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json not found after building artifacts.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", required=True)
    ap.add_argument("--y_path", required=True)
    ap.add_argument("--sheet_x", required=True)
    ap.add_argument("--sheet_y", required=True)
    ap.add_argument("--date_col", required=True)
    ap.add_argument("--target_col", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    metrics = run_all(
        x_path=args.x_path,
        y_path=args.y_path,
        sheet_x=args.sheet_x,
        sheet_y=args.sheet_y,
        date_col=args.date_col,
        target_col=args.target_col,
        out_dir=args.out_dir,
    )
    print("Done. Metrics:", metrics)


if __name__ == "__main__":
    main()
