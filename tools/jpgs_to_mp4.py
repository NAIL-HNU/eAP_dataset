#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = (
    "/home/jhang/workspace/nnTTC/nnttc_code_clean/dataset_generation/outputs/EnnTTC_v2/TRO_pse7/filtered_anno/TRO_pse7/eX8t2nGd6O"
)


def _natural_key(path: Path) -> list:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def _list_images(input_dir: Path, ext: str) -> list[Path]:
    return sorted(input_dir.glob(f"*.{ext}"), key=_natural_key)


def _run_ffmpeg(
    input_dir: Path,
    ext: str,
    output: Path,
    fps: int,
    crf: int,
    preset: str,
) -> int:
    pattern = str(input_dir / f"*.{ext}")
    cmd = [
        "ffmpeg",
        "-threads",
        "0",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    logging.info("ffmpeg cmd: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _run_opencv(image_files: list[Path], output: Path, fps: int) -> int:
    try:
        import cv2
    except Exception as exc:
        print(f"无法导入 cv2，OpenCV 写视频失败: {exc}", file=sys.stderr)
        return 2

    first = cv2.imread(str(image_files[0]))
    if first is None:
        print("无法读取第一张图像。", file=sys.stderr)
        return 2

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("无法创建 VideoWriter。", file=sys.stderr)
        return 2

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("跳过无法读取的图像: %s", img_path)
            continue
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        writer.write(img)

    writer.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="将目录内 JPG 图像合成为 MP4 视频。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help="输入 JPG 目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 mp4 路径（默认与输入目录同级，文件名为目录名.mp4）",
    )
    parser.add_argument("--ext", type=str, default="jpg", help="图像后缀名")
    parser.add_argument("--fps", type=int, default=10, help="输出帧率")
    parser.add_argument("--crf", type=int, default=28, help="x264 crf（ffmpeg 模式）")
    parser.add_argument("--preset", type=str, default="fast", help="x264 preset（ffmpeg 模式）")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "ffmpeg", "opencv"],
        help="合成方式：auto 优先 ffmpeg，否则 opencv",
    )
    parser.add_argument("--verbose", action="store_true", help="输出更多日志")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"输入目录不存在: {input_dir}", file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        output = input_dir.parent / f"{input_dir.name}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    image_files = _list_images(input_dir, args.ext)
    if not image_files:
        print(f"未找到 .{args.ext} 图像: {input_dir}", file=sys.stderr)
        return 1

    method = args.method
    if method == "auto":
        method = "ffmpeg" if shutil.which("ffmpeg") else "opencv"

    if method == "ffmpeg":
        if shutil.which("ffmpeg") is None:
            print("未找到 ffmpeg，可改用 --method opencv。", file=sys.stderr)
            return 2
        code = _run_ffmpeg(input_dir, args.ext, output, args.fps, args.crf, args.preset)
        if code != 0:
            print("ffmpeg 执行失败，可尝试 --method opencv。", file=sys.stderr)
        return code

    return _run_opencv(image_files, output, args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
