"""Run the pipeline over still images and print what it decided.

Phase 1 has no camera and no browser yet, so this is how the engine gets
exercised: point it at a photo, see the per-face verdicts and the numbers
behind them. It stays useful later as the quickest way to debug a frame the
live system got wrong.

    python -m engine.demo path/to/image.jpg [more.jpg ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from engine.gallery import Gallery
from engine.pipeline import PresencePipeline


def describe(path: Path, pipeline: PresencePipeline, repeat: int) -> None:
    image = cv2.imread(str(path))
    if image is None:
        print(f"!! could not read {path}")
        return

    # The tracker carries state between frames, so feeding the same still
    # repeatedly is how a static image reaches a stable identity - it
    # simulates the same face persisting across a run of frames.
    for _ in range(repeat):
        result = pipeline.process(image)

    print(f"\n{path.name}  ({result.frame_width}x{result.frame_height}, "
          f"{result.elapsed_ms:.0f}ms, {len(result.faces)} face(s))")
    print("-" * 76)

    if not result.faces:
        print("  no faces detected")
        return

    for face in sorted(result.faces, key=lambda f: f.bbox[0]):
        name = face.name or "-"
        print(
            f"  [ID {face.track_id:>2}] {face.state.value:<11} {name:<12}"
            f" sim={face.similarity:>6.3f} margin={face.margin:>6.3f}"
            f" blur={face.blur_score:>7.1f} det={face.det_score:.2f}"
        )
        if face.detail:
            print(f"{'':>13} {face.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="process each image N times to let temporal smoothing stabilise",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from engine.detector import FaceDetector

    detector = FaceDetector()
    gallery = Gallery.from_directory(detector=detector)
    print(f"gallery: {len(gallery)} enrolled -> {', '.join(gallery.names) or '(empty)'}")

    pipeline = PresencePipeline(detector=detector, gallery=gallery)

    for path in args.images:
        # Each image is an independent scene, so tracks must not carry over.
        from engine.tracker import FaceTracker

        pipeline.tracker = FaceTracker()
        describe(path, pipeline, args.repeat)

    return 0


if __name__ == "__main__":
    sys.exit(main())
