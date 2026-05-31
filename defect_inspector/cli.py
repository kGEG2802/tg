"""Інтерфейс командного рядка для defect_inspector."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import InspectionConfig
from .image_io import SUPPORTED_EXTS
from .pipeline import Inspector
from .report import write_reports
from .samples import write_demo_pair


def _build_inspector(args) -> Inspector:
    config = InspectionConfig.from_file(args.config) if args.config else InspectionConfig()
    return Inspector(config)


def _print_result(result, json_path=None, img_path=None) -> None:
    line = f"[{result.verdict}] {result.source} — дефектів: {len(result.defects)}, " \
           f"макс. тяжкість: {result.max_severity:.2f}, {result.elapsed_ms:.0f} мс"
    print(line)
    for d in result.defects:
        x, y, w, h = d.bbox
        print(f"    • {d.kind:<10} sev={d.severity:.2f} bbox=({x},{y},{w},{h})")
    if result.reasons:
        print("    причини FAIL: " + "; ".join(result.reasons))
    if json_path:
        print(f"    звіт: {json_path}")
    if img_path:
        print(f"    зображення: {img_path}")


# ----------------------------------------------------------------- commands
def cmd_inspect(args) -> int:
    inspector = _build_inspector(args)
    result = inspector.inspect(args.image, reference=args.reference)
    json_path = img_path = None
    if args.out:
        json_path, img_path = write_reports(
            args.image, result, args.out, inspector.config.preprocess
        )
    _print_result(result, json_path, img_path)
    return 0 if result.passed else 2


def cmd_batch(args) -> int:
    inspector = _build_inspector(args)
    in_dir = Path(args.input)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        print(f"У {in_dir} немає підтримуваних зображень.", file=sys.stderr)
        return 1

    passed = failed = 0
    for f in files:
        result = inspector.inspect(f, reference=args.reference)
        json_path = img_path = None
        if args.out:
            json_path, img_path = write_reports(
                f, result, args.out, inspector.config.preprocess
            )
        _print_result(result, json_path, img_path)
        passed += int(result.passed)
        failed += int(not result.passed)
    print(f"\nРазом: {len(files)} | PASS: {passed} | FAIL: {failed}")
    return 0 if failed == 0 else 2


def cmd_demo(args) -> int:
    out = Path(args.out)
    good, bad = write_demo_pair(out)
    print(f"Згенеровано зразки: {good}, {bad}\n")
    inspector = _build_inspector(args)
    for img, ref in [(bad, good if args.reference_mode else None), (good, None)]:
        result = inspector.inspect(img, reference=ref)
        json_path, img_path = write_reports(img, result, out, inspector.config.preprocess)
        _print_result(result, json_path, img_path)
        print()
    return 0


def cmd_init_config(args) -> int:
    InspectionConfig().save(args.path)
    print(f"Конфіг за замовчуванням збережено: {args.path}")
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="defect_inspector",
        description="Інспекція деталей на дефекти (комп'ютерний зір, офлайн).",
    )
    p.add_argument("--config", help="шлях до JSON-конфігу порогів")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect", help="перевірити одне зображення")
    pi.add_argument("image", help="шлях до зображення деталі")
    pi.add_argument("--reference", help="еталонне (годне) зображення для порівняння")
    pi.add_argument("--out", help="каталог для звіту JSON + анотованого зображення")
    pi.set_defaults(func=cmd_inspect)

    pb = sub.add_parser("batch", help="перевірити всі зображення в каталозі")
    pb.add_argument("input", help="каталог із зображеннями")
    pb.add_argument("--reference", help="єдиний еталон для всіх зображень")
    pb.add_argument("--out", help="каталог для звітів")
    pb.set_defaults(func=cmd_batch)

    pd = sub.add_parser("demo", help="згенерувати синтетичні зразки та перевірити їх")
    pd.add_argument("--out", default="demo_out", help="каталог результатів (типово demo_out)")
    pd.add_argument("--reference-mode", action="store_true",
                    help="перевіряти дефектну деталь у режимі порівняння з еталоном")
    pd.set_defaults(func=cmd_demo)

    pc = sub.add_parser("init-config", help="створити JSON-конфіг за замовчуванням")
    pc.add_argument("path", help="куди зберегти конфіг")
    pc.set_defaults(func=cmd_init_config)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
