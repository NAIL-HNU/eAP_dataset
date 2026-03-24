#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

from visualize_release_sequence import (
    EventStreamReader,
    choose_sequence_dir,
    compose_frame,
    is_sequence_dir,
    iter_sequence_dirs,
    load_sequence_frames,
)

DEFAULT_OUTPUT_DIR = Path("./release_checks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample rendered release frames for manual spot checks.")
    parser.add_argument("release_path", type=Path, help="Release root or a single sequence directory.")
    parser.add_argument("--sequence-id", type=str, default=None, help="Only check this sequence when passing the release root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for sampled PNG outputs.")
    parser.add_argument("--samples-per-asset", type=int, default=5, help="Number of random frames to export per asset.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for deterministic sampling.")
    parser.add_argument("--image-width", type=int, default=960, help="Width of each left-column panel.")
    parser.add_argument("--bev-width", type=int, default=760, help="Width of the BEV column.")
    parser.add_argument("--fwd-max", type=float, default=60.0, help="BEV forward range upper bound in meters.")
    parser.add_argument("--lat-max", type=float, default=30.0, help="BEV half lateral range in meters.")
    parser.add_argument("--pixel-diff", type=int, default=0, help="Extra x-axis pixel shift for event overlay.")
    return parser.parse_args()


def build_rng(sequence_id: str, seed: int) -> random.Random:
    token = f"{sequence_id}:{seed}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return random.Random(int(digest[:16], 16))


def choose_sequence_dirs(release_path: Path, sequence_id: str | None) -> List[Path]:
    if is_sequence_dir(release_path):
        return [choose_sequence_dir(release_path, sequence_id)]
    if sequence_id is not None:
        return [choose_sequence_dir(release_path, sequence_id)]
    return iter_sequence_dirs(release_path)


def render_samples_for_sequence(
    sequence_dir: Path,
    output_root: Path,
    samples_per_asset: int,
    seed: int,
    image_width: int,
    bev_width: int,
    fwd_max: float,
    lat_max: float,
    pixel_diff: int,
) -> Dict:
    frames = load_sequence_frames(sequence_dir)
    sequence_id = sequence_dir.name
    sample_count = min(max(1, samples_per_asset), len(frames))
    rng = build_rng(sequence_id, seed)
    selected_indices = sorted(rng.sample(range(len(frames)), k=sample_count))

    sequence_output_dir = output_root / sequence_id
    sequence_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    reader = EventStreamReader(frames[0].events_path)
    try:
        for sample_order, frame_index in enumerate(selected_indices, start=1):
            frame = frames[frame_index]
            composed = compose_frame(
                frame,
                reader,
                frame_index,
                len(frames),
                image_width=image_width,
                bev_width=bev_width,
                pixel_diff=pixel_diff,
                fwd_range=(0.0, float(fwd_max)),
                lat_range=(-float(lat_max), float(lat_max)),
            )
            frame_stem = Path(frame.file_name).stem
            image_name = f"{sample_order:02d}__{frame_stem}.png"
            image_path = sequence_output_dir / image_name
            if not cv2.imwrite(str(image_path), composed):
                raise RuntimeError(f"Failed to save sample image: {image_path}")
            manifest_rows.append(
                {
                    "sample_order": sample_order,
                    "frame_index": frame_index,
                    "file_name": frame.file_name,
                    "image_name": image_name,
                    "rgb_exposure_start_timestamp_us": frame.exposure_start_us,
                    "rgb_exposure_end_timestamp_us": frame.exposure_end_us,
                    "object_count": len(frame.objects),
                }
            )
    finally:
        reader.close()

    manifest = {
        "sequence_id": sequence_id,
        "seed": seed,
        "samples_per_asset": samples_per_asset,
        "selected_count": sample_count,
        "total_frames": len(frames),
        "samples": manifest_rows,
    }
    manifest_path = sequence_output_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    release_path = args.release_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dirs = choose_sequence_dirs(release_path, args.sequence_id)

    summaries = []
    for sequence_dir in tqdm(sequence_dirs, desc="Sampling assets"):
        summaries.append(
            render_samples_for_sequence(
                sequence_dir=sequence_dir,
                output_root=output_dir,
                samples_per_asset=args.samples_per_asset,
                seed=args.seed,
                image_width=args.image_width,
                bev_width=args.bev_width,
                fwd_max=args.fwd_max,
                lat_max=args.lat_max,
                pixel_diff=args.pixel_diff,
            )
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"assets": summaries}, ensure_ascii=False, indent=2) + "\n")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
