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
pre-recorded footage. Three rules give the randomness teeth: evidence only
counts after the prompt appears, the face must return to neutral between
challenges so one held pose cannot satisfy two, and every challenge has a
deadline so an attacker cannot cycle through poses until one lands.

**Liveness thresholds are relative to each person, not global.** Resting eye
openness varies enough between people that a fixed cutoff reads some faces as
permanently mid-blink. Each person's resting geometry is measured while their
identity is being confirmed — dead time the system already spends — and blink
and smile tests are expressed as fractions of it.

**Multiple reference images per person.** A single enrolment photo pins an
identity to one pose and one lighting condition, and everything that deviates
from it scores lower. Enrolment captures several frames across poses and stores
each as its own reference, rather than averaging them into one vector that
represents no pose particularly well.

**SQLite, not a CSV log.** Presence is a question with a shape — who is
enrolled, when did they arrive, were they here yesterday. A flat file answers
none of those without being parsed back into a database anyway, and it cannot
express that the same person passing the camera twice in an hour is one arrival
rather than two.

## Status

Working end to end. Enrol someone through the browser, stand in front of the
kiosk, answer the challenge, and the check-in lands in SQLite.

Still to come: the evaluation harness, and a performance pass — recognition
currently runs at roughly 5 fps for one face and slower for several, which is
usable but not yet comfortable.

## Running it

Requires Python 3.13.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m server.app
```

Then open <http://127.0.0.1:8000>. Model weights (~300MB) download on first run.
Locally no password is needed; the server only listens on this machine.

- **Enrol** captures you across five head positions and stores each as its own
  reference.
- **Kiosk** is the check-in screen: stand in front of it and answer the prompt.
- **History** lists every check-in with the confidence behind it.

The browser owns the camera and all rendering; the server owns the pipeline.
Frames are paced request-response — the page sends one frame and waits for its
verdict before capturing the next — so a slow frame delays the next capture
instead of building a backlog of frames that are already wrong by the time they
are processed.

There is also a headless path for debugging a specific image:

```bash
./.venv/bin/python -m engine.demo photo.jpg --repeat 6
```

## Deploying

The server needs a long-running process and a disk that survives restarts: the
gallery lives in SQLite, and the models should download once rather than on
every boot. That rules out serverless platforms such as Vercel, whose
filesystem is read-only and per-instance, so enrolments would vanish and
instances would disagree about who is enrolled.

A `Dockerfile` is included and runs on any container host with a volume. On
Railway:

1. Create a project from this GitHub repository. The Dockerfile is detected
   automatically.
2. Attach a volume mounted at `/data`. The database and model weights live
   there.
3. Set `PRESENCE_PASSWORD` to a long password, and `PRESENCE_SECRET` to a
   random string so sign-ins survive redeploys.
4. Generate a public domain, and set the health check path to `/healthz`.

The first boot downloads the models onto the volume, so it takes noticeably
longer than later ones.

**Every page, API call and websocket requires signing in.** A kiosk that answers
"who is this face?" with a name would otherwise be an identity oracle for anyone
holding the URL — hold up a photo of a stranger and learn whether they are
enrolled and what they are called. The server refuses to listen on a public
interface at all unless `PRESENCE_PASSWORD` is set.

| Variable | Purpose |
|---|---|
| `PRESENCE_PASSWORD` | Required for any non-local deployment |
| `PRESENCE_SECRET` | Signs session cookies; without it, devices sign in again after each restart |
| `PRESENCE_DATA_DIR` | Database location (the image sets `/data`) |
| `PRESENCE_MODEL_DIR` | Model weights location (the image sets `/data/models`) |
| `PORT` | Set by the hosting platform |

## Tests

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest tests/ -q
```

The suite covers the decision logic — matching, rejection, tracking, smoothing
and the liveness state machine — using synthetic embeddings and geometry, so it
runs in a fraction of a second and needs no model or camera.

The liveness tests are written around the properties an attacker would try to
violate rather than the happy path: that a blink recorded before the prompt
does not count, that one long blink counts once rather than many times, that a
single held pose cannot satisfy two consecutive challenges, and that the
sequence is genuinely unpredictable.

`tests/test_integration.py` runs the whole pipeline over real photographs,
simulating a cooperative user by responding to whichever challenge it is
actually asked. It proves both directions: a person who answers is verified and
logged, and a static photo is recognised perfectly well yet never logged. It
needs photographs, which are personal data and are not committed, so it skips
unless `PRESENCE_TEST_FRAMES` points at a directory of them — see that module's
docstring.

## Layout

```
config.py     every tunable threshold, in one place
Dockerfile    container image for deployment
engine/       detection, matching, tracking, gating, liveness, storage — no UI
tools/        threshold calibration
server/       FastAPI kiosk: websocket frames in, verdicts out
web/          browser UI, canvas overlay
eval/         labelled set and metrics
tests/        decision-logic tests, plus an end-to-end integration test
```

## Calibration

Several thresholds are meaningless as absolute numbers. Laplacian variance
scales with crop resolution and camera; resting eye openness varies from person
to person. A value copied from a paper describes that paper's camera and
subjects, not yours.

```bash
./.venv/bin/python -m tools.calibrate --camera 0 --seconds 30
```

Sample yourself at a normal distance, blinking naturally and moving enough to
produce some genuinely blurred frames. It reports the distribution of every
measurement and recommends thresholds from what it saw.

This matters more than it sounds. The blur threshold was initially guessed at
60; measured against real footage the median frame scores 71, so that guess
would have rejected roughly half of all normal frames. Values still marked
PROVISIONAL in `config.py` have not yet been separated from ordinary movement
by a labelled pass.

## Built with

InsightFace on ONNX Runtime, OpenCV, FastAPI.

The buffalo_l pack supplies detection, 512-d embeddings, dense landmarks and
head pose from one forward pass, so liveness geometry costs no extra inference
and needs no second landmark library.
