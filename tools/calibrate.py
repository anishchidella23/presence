"""Measure the thresholds in config.py against a real camera or recording.

Several values in this system are meaningless as absolutes. Laplacian variance
scales with crop resolution, lens and sensor; resting eye openness varies from
person to person. A number copied from a paper describes that paper's camera
and that paper's subjects, not yours.

This tool samples frames, reports the distribution of each measurement, and
recommends thresholds derived from what it saw.

    python -m tools.calibrate --camera 0 --seconds 30
    python -m tools.calibrate --video clip.mp4
    python -m tools.calibrate --images 'frames/*.jpg'

While it runs on a camera, behave the way a user would: sit at a normal
distance, look around, blink naturally, and move enough to produce some
genuinely blurred frames.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

import cv2
import numpy as np

from engine import liveness
from engine.detector import FaceDetector, blur_score


def percentiles(values: list[float], points=(1, 5, 10, 50, 90, 99)) -> dict[int, float]:
    return {p: float(np.percentile(values, p)) for p in points} if values else {}


def summarise(label: str, values: list[float], unit: str = "") -> None:
    if not values:
        print(f"  {label}: no samples")
        return
    p = percentiles(values)
    print(f"  {label:<16} min={min(values):7.3f}{unit}  p05={p[5]:7.3f}  "
          f"median={p[50]:7.3f}  p95={p[90]:7.3f}  max={max(values):7.3f}{unit}")


def frames_from(args) -> tuple:
    """Yield BGR frames from whichever source was requested."""
    if args.images:
        paths = sorted(glob.glob(args.images))
        if not paths:
            sys.exit(f"no images matched {args.images!r}")
        return ((cv2.imread(p) for p in paths), len(paths))

    source = args.video if args.video else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        sys.exit(f"could not open source {source!r}")

    def generate():
        started = time.time()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.video is None and time.time() - started > args.seconds:
                break
            yield frame
        capture.release()

    return (generate(), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="camera index")
    source.add_argument("--video", type=str, help="video file instead of a camera")
    source.add_argument("--images", type=str, help="glob of still frames")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="how long to sample when reading a camera")
    parser.add_argument("--every", type=int, default=1, help="process every Nth frame")
    args = parser.parse_args(argv)

    detector = FaceDetector()
    stream, total = frames_from(args)

    blur: list[float] = []
    openness: list[float] = []
    yaw: list[float] = []
    smile: list[float] = []
    seen = matched = 0

    for index, frame in enumerate(stream):
        if frame is None or index % args.every:
            continue
        seen += 1
        for detection in detector.detect(frame):
            if detection.embedding is None:
                continue    # failed the size or score gate; not a usable sample
            matched += 1
            blur.append(blur_score(frame, detection.bbox))
            metrics = liveness.measure(
                detection.landmarks_2d, detection.keypoints, detection.pose
            )
            if metrics.valid:
                openness.append(metrics.eye_openness)
                smile.append(metrics.smile_ratio)
                yaw.append(metrics.yaw)

    print(f"\nsampled {seen} frames, {matched} usable face detections")
    if not blur:
        sys.exit("no usable faces found - check lighting, framing and distance")

    print("\ndistributions")
    summarise("blur (Laplacian)", blur)
    summarise("eye openness", openness)
    summarise("smile ratio", smile)
    summarise("yaw", yaw, "deg")

    # --- recommendations -------------------------------------------------
    # Blur: sit below the bulk of the distribution so ordinary frames pass and
    # only genuinely degraded ones are rejected. The 5th percentile is a
    # deliberately forgiving choice - a blur gate that is too eager is worse
    # than one slightly too lax, because rejected frames stall the user with
    # no way to tell what is wrong.
    blur_recommendation = np.percentile(blur, 5)

    print("\nrecommended config.py values")
    print(f"  BLUR_THRESHOLD = {blur_recommendation:.0f}")

    if openness:
        rest = float(np.median(openness))
        deepest = float(np.min(openness))
        print(f"  # resting eye openness {rest:.3f}, deepest observed {deepest:.3f}")
        if deepest < rest * 0.65:
            print("  BLINK_CLOSE_RATIO = 0.65   # blinks land well below this; unchanged")
        else:
            suggested = max(0.5, deepest / rest + 0.05)
            print(f"  BLINK_CLOSE_RATIO = {suggested:.2f}   # no deep blink seen; "
                  "loosened, re-run while blinking deliberately")

    if yaw:
        reach = max(abs(min(yaw)), abs(max(yaw)))
        if reach < 18.0:
            print(f"  # yaw only reached {reach:.0f} deg - turn your head further "
                  "when calibrating, or lower YAW_TURN_DEGREES")
        else:
            print(f"  # yaw reached {reach:.0f} deg, YAW_TURN_DEGREES = 18 is reachable")

    return 0


if __name__ == "__main__":
    sys.exit(main())
