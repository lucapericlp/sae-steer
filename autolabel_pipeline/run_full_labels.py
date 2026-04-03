from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


TOTAL_FEATURES = 10240
DEFAULT_SOURCE_RUN = Path("autolabel_pipeline/runs/benchmark_1024_bs160_32")
DEFAULT_FINAL_RUN = Path("autolabel_pipeline/runs/full_10k_bs160_32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume-friendly orchestration for the full 10,240-feature SAE autolabelling run.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--final-run", type=Path, default=DEFAULT_FINAL_RUN)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--generate-batch-size", type=int, default=160)
    parser.add_argument("--label-batch-size", type=int, default=32)
    parser.add_argument("--steering-scale", type=float, default=120.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 37, 89])
    parser.add_argument("--start-feature", type=int, default=0)
    parser.add_argument("--end-feature", type=int, default=TOTAL_FEATURES - 1)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copytree_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst)


def initialize_final_run(source_run: Path, final_run: Path) -> None:
    ensure_dir(final_run)
    images_dst = final_run / "images"
    images_src = source_run / "images"
    if images_src.exists():
        if not images_dst.exists():
            shutil.copytree(images_src, images_dst)
        else:
            for feature_dir in images_src.glob("feature_*"):
                dst = images_dst / feature_dir.name
                if not dst.exists():
                    shutil.copytree(feature_dir, dst)

    for name in ["labels.json", "labels.csv", "generation_manifest.json", "gpu_profile.jsonl", "stage_summary.jsonl"]:
        src = source_run / name
        dst = final_run / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def load_completed_feature_ids(final_run: Path) -> set[int]:
    labels_path = final_run / "labels.json"
    if not labels_path.exists():
        return set()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return {int(row["feature_id"]) for row in labels}


def append_jsonl(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    with dst.open("a", encoding="utf-8") as out_handle:
        out_handle.write(src.read_text(encoding="utf-8"))


def merge_chunk(chunk_dir: Path, final_run: Path) -> None:
    for feature_dir in (chunk_dir / "images").glob("feature_*"):
        dst = final_run / "images" / feature_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(feature_dir, dst)

    final_labels_path = final_run / "labels.json"
    final_manifest_path = final_run / "generation_manifest.json"

    final_labels = []
    final_manifest = {
        "feature_runs": [],
    }
    if final_labels_path.exists():
        final_labels = json.loads(final_labels_path.read_text(encoding="utf-8"))
    if final_manifest_path.exists():
        final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))

    chunk_labels = json.loads((chunk_dir / "labels.json").read_text(encoding="utf-8"))
    chunk_manifest = json.loads((chunk_dir / "generation_manifest.json").read_text(encoding="utf-8"))

    label_map = {int(row["feature_id"]): row for row in final_labels}
    for row in chunk_labels:
        label_map[int(row["feature_id"])] = row
    merged_labels = [label_map[key] for key in sorted(label_map)]
    final_labels_path.write_text(json.dumps(merged_labels, indent=2), encoding="utf-8")

    run_map = {}
    for row in final_manifest.get("feature_runs", []):
        run_map[(int(row["feature_id"]), int(row["seed"]))] = row
    for row in chunk_manifest["feature_runs"]:
        run_map[(int(row["feature_id"]), int(row["seed"]))] = row

    for key in ["model_id", "sae_repo_id", "sae_filename", "steering_scale", "height", "width", "generate_batch_size"]:
        if key in chunk_manifest:
            final_manifest[key] = chunk_manifest[key]
    final_manifest["feature_runs"] = [run_map[key] for key in sorted(run_map)]
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")

    csv_path = final_run / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_id", "label", "reason"])
        writer.writeheader()
        for row in merged_labels:
            writer.writerow({key: row[key] for key in ["feature_id", "label", "reason"]})

    append_jsonl(chunk_dir / "gpu_profile.jsonl", final_run / "gpu_profile.jsonl")
    append_jsonl(chunk_dir / "stage_summary.jsonl", final_run / "stage_summary.jsonl")


def chunk_ranges(feature_ids: list[int], chunk_size: int) -> list[list[int]]:
    return [feature_ids[index : index + chunk_size] for index in range(0, len(feature_ids), chunk_size)]


def is_generation_complete(chunk_dir: Path, chunk_ids: list[int], seeds: list[int]) -> bool:
    manifest_path = chunk_dir / "generation_manifest.json"
    if not manifest_path.exists():
        return False
    expected_image_count = len(chunk_ids) * len(seeds)
    actual_image_count = sum(1 for _ in (chunk_dir / "images").rglob("*.png")) if (chunk_dir / "images").exists() else 0
    if actual_image_count != expected_image_count:
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return len(manifest.get("feature_runs", [])) == expected_image_count


def run_chunk(chunk_ids: list[int], chunk_dir: Path, args: argparse.Namespace) -> None:
    if is_generation_complete(chunk_dir, chunk_ids, args.seeds):
        cmd = [
            sys.executable,
            str(Path("autolabel_pipeline/autolabel.py").resolve()),
            "label",
            "--output-dir",
            str(chunk_dir),
            "--feature-ids",
            *[str(feature_id) for feature_id in chunk_ids],
            "--max-new-tokens",
            "96",
            "--label-batch-size",
            str(args.label_batch_size),
            "--cpu-workers",
            "8",
        ]
        subprocess.run(cmd, check=True)
        return

    cmd = [
        sys.executable,
        str(Path("autolabel_pipeline/autolabel.py").resolve()),
        "run",
        "--output-dir",
        str(chunk_dir),
        "--feature-ids",
        *[str(feature_id) for feature_id in chunk_ids],
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--steering-scale",
        str(args.steering_scale),
        "--generate-batch-size",
        str(args.generate_batch_size),
        "--label-batch-size",
        str(args.label_batch_size),
        "--cpu-workers",
        "8",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    initialize_final_run(args.source_run, args.final_run)

    completed = load_completed_feature_ids(args.final_run)
    target_ids = list(range(args.start_feature, args.end_feature + 1))
    remaining = [feature_id for feature_id in target_ids if feature_id not in completed]

    print(
        json.dumps(
            {
                "completed_feature_count": len(completed),
                "remaining_feature_count": len(remaining),
                "source_run": str(args.source_run.resolve()),
                "final_run": str(args.final_run.resolve()),
            },
            indent=2,
        )
    )

    chunks_root = args.final_run / "chunk_runs"
    ensure_dir(chunks_root)

    for chunk_index, chunk_ids in enumerate(chunk_ranges(remaining, args.chunk_size), start=1):
        chunk_dir = chunks_root / f"chunk_{chunk_index:03d}_{chunk_ids[0]}_{chunk_ids[-1]}"
        if (chunk_dir / "labels.json").exists():
            merge_chunk(chunk_dir, args.final_run)
            continue
        run_chunk(chunk_ids, chunk_dir, args)
        merge_chunk(chunk_dir, args.final_run)

    final_labels = json.loads((args.final_run / "labels.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "final_label_count": len(final_labels),
                "expected_label_count": args.end_feature - args.start_feature + 1,
                "final_run": str(args.final_run.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
