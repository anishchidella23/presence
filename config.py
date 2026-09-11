"""Central configuration for the presence verification system.

Every tunable lives here so thresholds can be calibrated in one place rather
than hunted down across modules. Values marked PROVISIONAL are starting points
that get replaced by measured values once the evaluation harness exists.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FACES_DIR = DATA_DIR / "faces"        # enrolled reference images, per person
CAPTURES_DIR = DATA_DIR / "captures"  # frames saved during enrollment
DB_PATH = DATA_DIR / "presence.db"
MODEL_ROOT = ROOT / "models"


# --- Detection / recognition model --------------------------------------

# buffalo_l bundles RetinaFace detection with an ArcFace r100 recogniser and
# produces 512-d embeddings. Downloaded once on first run.
MODEL_PACK = "buffalo_l"
DET_SIZE = (640, 640)

# Detections below this score are noise - usually background texture or a
# partially occluded face at the frame edge.
MIN_DETECTION_SCORE = 0.60

# A face smaller than this (in full-frame pixels, longest bbox edge) carries
# too little detail to embed reliably. The original system's "far distance
# degrades to Unknown" finding is really this constraint showing up untreated.
MIN_FACE_PIXELS = 80


# --- Identity matching ---------------------------------------------------

# Embeddings are L2-normalised, so a dot product is cosine similarity in
# [-1, 1]. Same person typically scores well above 0.5; different people
# usually below 0.35, though degraded images (dim, off-angle, photographed
# from a screen) push impostor pairs higher.
#
# PROVISIONAL - calibrate against the labelled evaluation set.
RECOGNITION_THRESHOLD = 0.45

# Open-set safeguard: the best match must beat the runner-up by this margin.
# An absolute threshold alone is fragile because the right cutoff drifts with
# lighting and camera. If the top candidate is not clearly better than the
# second, we genuinely do not know who this is and should say Unknown rather
# than force a match.
#
# PROVISIONAL - calibrate against the labelled evaluation set.
MIN_MATCH_MARGIN = 0.08


# --- Frame quality gates -------------------------------------------------

# Variance of the Laplacian over the grayscale face crop. Sharp faces show
# strong edge variation; motion-blurred ones do not.
#
# This value is meaningless as an absolute - it scales with crop resolution
# and camera, so it MUST be recalibrated per camera (tools/calibrate.py).
#
# Measured over 220 frames of 720p handheld webcam footage: median 71,
# 5th percentile 30. An earlier guess of 60 would have rejected roughly half
# of ordinary frames. A blur gate that is too eager is worse than one slightly
# too lax, because a rejected frame stalls the user without telling them why.
BLUR_THRESHOLD = 30.0


# --- Temporal smoothing --------------------------------------------------

# A single frame should never decide an identity. The same name must win a
# majority of a short rolling window before the track is considered stable.
SMOOTHING_WINDOW = 10
REQUIRED_MATCHES = 5


# --- Tracking ------------------------------------------------------------

# Minimum IoU for a detection to continue an existing track.
TRACK_IOU_THRESHOLD = 0.30

# How many consecutive frames a track may go unmatched before it is dropped.
# Tolerates brief detector dropouts without losing accumulated liveness state.
TRACK_MAX_AGE = 15


# --- Liveness geometry ---------------------------------------------------

# Landmarks sampled around each eye keypoint to measure openness. Enough to
# span the eyelids and corners without reaching into the brow or cheek.
EYE_LANDMARK_COUNT = 10

# Frames of neutral geometry collected before challenges begin. Gathered
# during identification, so this costs the user no extra waiting.
BASELINE_SAMPLES = 12

# Blink thresholds are fractions of a person's own resting eye openness, not
# absolute values: resting openness varies enough between people that a fixed
# cutoff reads some faces as permanently mid-blink.
#
# Measured: resting openness 0.351, a real blink bottoms out at 0.117 - a third
# of resting. 0.65 sits comfortably between them.
#
# The gap between the two makes the test hysteretic - eyes must fall below the
# close ratio and then rise past the higher open ratio to count as one blink,
# so a value hovering at the boundary cannot register a burst of phantom
# blinks.
BLINK_CLOSE_RATIO = 0.65
BLINK_OPEN_RATIO = 0.85

# Mouth width must exceed this multiple of the person's neutral width to count
# as a smile.
#
# PROVISIONAL. Calibration footage reached 1.21x neutral, but it contains
# speech and no deliberate smiling, so the ceiling of ordinary mouth movement
# is not yet separated from a real smile. Set above the observed range for now;
# needs a labelled pass to tighten.
SMILE_RISE_RATIO = 1.18

# Degrees of yaw that count as a deliberate head turn.
#
# Measured: incidental movement while seated stays within +/-7 degrees, while a
# deliberate turn reaches 40. 18 sits clearly between the two.
YAW_TURN_DEGREES = 18.0


# --- Challenge sequence --------------------------------------------------

# How many challenges must be passed. Each one multiplies the difficulty of
# anticipating the sequence in advance, which is what defeats replayed video.
CHALLENGE_COUNT = 2

# Seconds allowed per challenge before the attempt fails. Long enough to read
# the prompt and respond, short enough that cycling through poses hoping to
# hit the right one is not viable.
CHALLENGE_TIMEOUT_S = 10.0

# Upper bound on the randomised blink repeat count.
MAX_BLINK_REPEATS = 3
