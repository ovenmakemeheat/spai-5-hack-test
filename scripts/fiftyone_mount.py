#!/usr/bin/env python
"""Quickly mount local image and tabular datasets in FiftyOne."""

from __future__ import annotations

import argparse
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

PLACEHOLDER_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

RESERVED_FIELDS = {"filepath", "id", "metadata", "tags"}


def import_fiftyone():
    try:
        import fiftyone as fo
    except ImportError as exc:
        raise SystemExit(
            "fiftyone is not installed. Install it with: uv add fiftyone --no-sync && uv sync"
        ) from exc

    return fo


def default_name(path: Path, prefix: str) -> str:
    clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem or path.name).strip("-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{clean_stem}-{timestamp}"


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() and root is not None:
        value = root / value
    return value.resolve()


def ensure_placeholder_image(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    placeholder = work_dir / "blank.png"
    if not placeholder.exists():
        placeholder.write_bytes(base64.b64decode(PLACEHOLDER_PNG))
    return placeholder.resolve()


def safe_field_name(name: str, used: set[str]) -> str:
    field = re.sub(r"\W+", "_", str(name)).strip("_").lower()
    if not field:
        field = "field"
    if field[0].isdigit():
        field = f"col_{field}"
    if field in RESERVED_FIELDS:
        field = f"{field}_value"

    candidate = field
    suffix = 2
    while candidate in used:
        candidate = f"{field}_{suffix}"
        suffix += 1

    used.add(candidate)
    return candidate


def normalize_value(value: Any) -> Any:
    if is_missing(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def read_table(path: Path, limit: int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=limit)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=limit)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise SystemExit(f"Unsupported tabular format: {suffix}")

    if limit is not None:
        return df.head(limit)
    return df


def detect_media_column(df: pd.DataFrame) -> str | None:
    preferred = ["filepath", "file_path", "path", "image", "image_path", "filename", "file_name"]
    columns_by_lower = {str(column).lower(): column for column in df.columns}
    for name in preferred:
        column = columns_by_lower.get(name)
        if column is not None and column_looks_like_image_paths(df[column]):
            return str(column)

    for column in df.columns:
        if column_looks_like_image_paths(df[column]):
            return str(column)

    return None


def column_looks_like_image_paths(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(50)
    if values.empty:
        return False
    matches = sum(Path(value).suffix.lower() in IMAGE_EXTENSIONS for value in values)
    return matches >= max(1, len(values) // 2)


def resolve_media_path(value: Any, media_root: Path | None, fallback: Path) -> Path:
    if is_missing(value) or str(value).strip() == "":
        return fallback

    path = Path(str(value)).expanduser()
    if not path.is_absolute() and media_root is not None:
        path = media_root / path
    return path.resolve()


def launch_dataset(fo: Any, dataset: Any, args: argparse.Namespace) -> None:
    print(dataset)
    if args.head:
        print(dataset.head(args.head))

    if args.no_app:
        return

    session = fo.launch_app(
        dataset,
        address=args.address,
        port=args.port,
        remote=args.remote,
        auto=not args.no_browser,
    )
    print(f"FiftyOne App launched for dataset '{dataset.name}'")
    if args.wait:
        session.wait()


def mount_images(args: argparse.Namespace) -> None:
    fo = import_fiftyone()
    source = Path(args.path).expanduser()
    name = args.name or default_name(source, "images")

    if args.overwrite and fo.dataset_exists(name):
        fo.delete_dataset(name)

    if source.exists() and source.is_dir():
        dataset = fo.Dataset.from_images_dir(
            str(source.resolve()),
            name=name,
            persistent=args.persistent,
            progress=False,
        )
    else:
        dataset = fo.Dataset.from_images_patt(
            str(source),
            name=name,
            persistent=args.persistent,
            progress=False,
        )

    launch_dataset(fo, dataset, args)


def mount_table(args: argparse.Namespace) -> None:
    fo = import_fiftyone()
    table_path = resolve_path(args.path)
    if not table_path.exists():
        raise SystemExit(f"Table not found: {table_path}")

    name = args.name or default_name(table_path, "table")
    if args.overwrite and fo.dataset_exists(name):
        fo.delete_dataset(name)

    df = read_table(table_path, args.limit)
    if df.empty:
        raise SystemExit(f"No rows found in {table_path}")

    media_root = resolve_path(args.media_root) if args.media_root else table_path.parent
    media_col = args.media_col or detect_media_column(df)
    placeholder = ensure_placeholder_image(Path(args.work_dir))

    used_fields = {"row_index"}
    field_map = {
        column: safe_field_name(str(column), used_fields)
        for column in df.columns
        if str(column) != str(media_col)
    }

    dataset = fo.Dataset(name=name, persistent=args.persistent)
    dataset.info["source_table"] = str(table_path)
    dataset.info["media_column"] = media_col
    dataset.info["field_map"] = {str(key): value for key, value in field_map.items()}

    samples = []
    for row_index, row in df.iterrows():
        filepath = (
            resolve_media_path(row[media_col], media_root, placeholder)
            if media_col is not None
            else placeholder
        )
        sample = fo.Sample(filepath=str(filepath))
        sample["row_index"] = int(row_index)
        for column, field in field_map.items():
            value = normalize_value(row[column])
            if value is not None:
                sample[field] = value
        samples.append(sample)

        if len(samples) >= args.batch_size:
            dataset.add_samples(samples, dynamic=True, progress=False)
            samples.clear()

    if samples:
        dataset.add_samples(samples, dynamic=True, progress=False)

    launch_dataset(fo, dataset, args)


def add_launch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", help="FiftyOne dataset name")
    parser.add_argument("--persistent", action="store_true", help="Persist dataset in FiftyOne DB")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing dataset with --name",
    )
    parser.add_argument("--port", type=int, default=5151, help="FiftyOne App port")
    parser.add_argument("--address", default="localhost", help="FiftyOne App bind address")
    parser.add_argument("--remote", action="store_true", help="Launch a remote FiftyOne session")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser")
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Create and print dataset without launching app",
    )
    parser.add_argument("--wait", action="store_true", help="Block until the app session is closed")
    parser.add_argument("--head", type=int, default=3, help="Print the first N samples")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quickly mount image folders/globs or tabular files in FiftyOne."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    images = subparsers.add_parser("images", help="Mount an image directory or glob")
    images.add_argument("path", help="Image directory or glob pattern")
    add_launch_args(images)
    images.set_defaults(func=mount_images)

    table = subparsers.add_parser("table", help="Mount CSV/TSV/Parquet/JSON rows")
    table.add_argument("path", help="Tabular file path")
    table.add_argument("--media-col", help="Column containing image file paths")
    table.add_argument(
        "--media-root",
        help="Root for relative media paths; defaults to table directory",
    )
    table.add_argument("--limit", type=int, help="Only load the first N rows")
    table.add_argument("--batch-size", type=int, default=500, help="Samples to insert per batch")
    table.add_argument("--work-dir", default=".fiftyone_mount", help="Directory for helper assets")
    add_launch_args(table)
    table.set_defaults(func=mount_table)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
