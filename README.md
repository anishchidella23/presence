# Presence

Real-time presence verification from a webcam. It recognises who is in front of
the camera, and — more importantly — decides whether it should believe itself
before recording anything.

Face recognition on its own is not presence verification. A recogniser will
happily put a name to a photograph held up to the lens, to a face too far away
to resolve, or to a single motion-blurred frame that happened to land near
someone in the gallery. This system treats a name as a hypothesis and puts it
through a series of gates, each of which can reject it:

```
detect ── quality ── identify ── smooth over time ── liveness challenge ── log
```

Presence is recorded only if a face clears every one.

## Design decisions worth explaining

**Open-set matching uses a margin, not just a threshold.** Deciding whether an
embedding belongs to an enrolled person is an open-set problem: the person at
the camera may not be enrolled at all. A lone similarity cutoff handles this
badly, because the right cutoff moves with lighting, camera and distance, so
any fixed value is too loose in good conditions or too tight in poor ones. This
system also requires the best candidate to beat the runner-up by a margin. If
the top match is not clearly better than the second, the embedding is not
landing distinctly on anyone, and the honest answer is `Unknown`. It is a
second, largely independent piece of evidence for roughly no extra cost.

**The engine returns data; it never draws.** Every stage produces plain
structures, and rendering happens in the browser. That keeps the whole pipeline
testable against still images with no camera and no UI attached, which is what
makes an automated evaluation possible at all.

**Tracking comes before gating, because every gate is stateful.** Temporal
smoothing needs to know what a particular face looked like over the last N
frames; a liveness challenge needs to know the person asked to look left is the
one who then looks left. Neither is expressible without an identity that
survives between frames.

**Liveness is a randomised challenge, not a blink.** Blink detection is
defeated by replaying a video of someone blinking. A challenge chosen at random
at request time — look left, blink twice, smile — cannot be satisfied by
pre-recorded footage.

**Multiple reference images per person.** A single enrolment photo pins an
identity to one pose and one lighting condition, and everything that deviates
from it scores lower. Enrolment captures several frames across poses.

## Status

Working: detection, embedding, gallery matching with the margin rule,
multi-face tracking with stable IDs, quality gates, temporal smoothing.

In progress: liveness challenges, SQLite persistence, the web kiosk, and the
evaluation harness.

## Running it

Requires Python 3.13.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Enrol someone by dropping reference images into `data/faces/`, either one image
per person or a folder per person:

```
data/faces/Alice/front.jpg
data/faces/Alice/left.jpg
data/faces/Bob.jpg
```

Then run the pipeline over a still image:

```bash
./.venv/bin/python -m engine.demo photo.jpg --repeat 6
```

`--repeat` feeds the same frame several times so temporal smoothing can reach a
stable identity, simulating a face persisting across a run of frames.

Model weights (~300MB) download automatically on first run.

## Tests

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest tests/ -q
```

The suite covers the decision logic — matching, rejection, tracking, smoothing
— using synthetic embeddings, so it runs in a fraction of a second and needs no
model or camera.

## Layout

```
config.py     every tunable threshold, in one place
engine/       detection, matching, tracking, gating — no UI
server/       FastAPI kiosk: websocket frames in, verdicts out
web/          browser UI, canvas overlay
eval/         labelled set and metrics
tests/        decision-logic tests
```

## Calibration

Thresholds in `config.py` marked PROVISIONAL are starting points, not measured
values. The blur threshold in particular is meaningless as an absolute number:
Laplacian variance scales with crop resolution and camera, so it has to be
calibrated per setup. The evaluation harness replaces these with measured
values.

## Built with

InsightFace (RetinaFace detection, ArcFace embeddings) on ONNX Runtime,
MediaPipe face mesh for liveness geometry, OpenCV, FastAPI.
