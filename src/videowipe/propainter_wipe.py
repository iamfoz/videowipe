"""Run a user-provided ProPainter checkout as an external VideoWipe model.

VideoWipe does not distribute ProPainter code or weights. Install the optional
``videowipe[propainter]`` dependencies, then pass ``--propainter-dir`` or set
``VIDEOWIPE_PROPINTER_DIR`` to a separately licensed checkout.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


def _resolve_propainter_dir(arg=None):
    if arg:
        return os.fspath(arg)
    env = os.environ.get("VIDEOWIPE_PROPINTER_DIR")
    if env:
        return env
    raise ValueError(
        "ProPainter checkout is not configured; pass --propainter-dir or set "
        "VIDEOWIPE_PROPINTER_DIR"
    )


def _load_media_dependencies():
    try:
        import cv2
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "The ProPainter adapter requires videowipe[propainter]"
        ) from exc
    return cv2, imageio


def extract_frames(video_path, frames_dir):
    """Extract frames from video using imageio and return its fps."""
    cv2, imageio = _load_media_dependencies()
    reader = imageio.get_reader(video_path, "ffmpeg")
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps", 24)
        for index, frame in enumerate(reader):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(frames_dir, f"{index:06d}.png"), bgr)
    finally:
        reader.close()
    return fps


def frames_to_video(frames_dir, output_path, fps):
    """Combine rendered frames into a video."""
    _, imageio = _load_media_dependencies()
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    frames = [imageio.imread(path) for path in paths]
    imageio.mimwrite(output_path, frames, fps=fps, quality=7)
    print(f"Encoded {len(frames)} frames -> {output_path}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run ProPainter as a VideoWipe external model adapter.",
    )
    parser.add_argument("video_path")
    parser.add_argument("mask_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--propainter-dir",
        default=None,
        help="Path to a ProPainter checkout (or set VIDEOWIPE_PROPINTER_DIR)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional file for ProPainter stdout/stderr (discarded by default)",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        propainter_dir = _resolve_propainter_dir(args.propainter_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    inference_script = os.path.join(propainter_dir, "inference_propainter.py")
    if not os.path.isfile(inference_script):
        print(
            f"ERROR: ProPainter not found at {propainter_dir}. "
            "Pass --propainter-dir or set VIDEOWIPE_PROPINTER_DIR.",
            file=sys.stderr,
        )
        return 1

    video_path = os.path.abspath(args.video_path)
    mask_path = os.path.abspath(args.mask_path)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isfile(video_path):
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(mask_path):
        print(f"ERROR: mask not found: {mask_path}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="propainter_")
    try:
        frames_dir = os.path.join(work, "frames")
        os.makedirs(frames_dir)
        fps = extract_frames(video_path, frames_dir)

        pp_tmp = os.path.join(work, "pp_output")
        os.makedirs(pp_tmp)
        command = [
            sys.executable,
            inference_script,
            "-i", frames_dir,
            "-m", mask_path,
            "-o", pp_tmp,
            "--fp16",
            # ProPainter only accepts a whole number here and uses it for its own
            # inpaint_out.mp4. The frames are saved as well and encoded below at
            # the source's real rate (29.97 would otherwise become 29 or 30).
            "--save_fps", str(max(1, round(fps))),
            "--save_frames",
        ]
        log_handle = None
        output_sink = subprocess.DEVNULL
        if args.log_file:
            try:
                log_handle = open(args.log_file, "w", encoding="utf-8")
            except OSError:
                print("ERROR: unable to open the requested log file", file=sys.stderr)
                return 1
            output_sink = log_handle
        try:
            result = subprocess.run(
                command,
                cwd=propainter_dir,
                stdout=output_sink,
                stderr=output_sink,
                check=False,
            )
        finally:
            if log_handle is not None:
                log_handle.close()
        if result.returncode != 0:
            print(
                f"ERROR: ProPainter exited with code {result.returncode}",
                file=sys.stderr,
            )
            if args.log_file:
                print(f"Details written to: {args.log_file}", file=sys.stderr)
            return 1

        # ProPainter names its result folder after its input, which is the
        # frames directory; the source's own name is kept for other versions.
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        results = [os.path.join(pp_tmp, name)
                   for name in (os.path.basename(frames_dir), video_name)]
        destination = os.path.join(output_dir, "inpaint_out.mp4")
        frames = next((os.path.join(root, "frames") for root in results
                       if os.path.isdir(os.path.join(root, "frames"))), None)
        pp_output = next((os.path.join(root, "inpaint_out.mp4") for root in results
                          if os.path.isfile(os.path.join(root, "inpaint_out.mp4"))), None)
        if frames is not None:
            frames_to_video(frames, destination, fps)
        elif pp_output is not None:
            if abs(fps - round(fps)) > 1e-3:
                print(f"Warning: ProPainter saved no frames; its video runs at "
                      f"{max(1, round(fps))} fps instead of {fps:.3f}.")
            shutil.copy2(pp_output, destination)
        else:
            print("ERROR: ProPainter output not found", file=sys.stderr)
            return 1
        print(f"Output: {destination}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
