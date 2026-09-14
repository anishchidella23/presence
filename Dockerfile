# Presence kiosk server.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV links against libGL and GLib even though nothing is ever displayed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Model weights are built into the image rather than fetched on first boot.
#
# Fetching at runtime needs the ~288MB zip and its ~325MB extraction on disk at
# once, which overflows small volumes such as Railway's 0.5GB trial limit. It
# also fails badly: insightface treats any existing model directory as a
# completed download, so a half-finished extraction left on a volume crashes
# every restart that follows. Done here, a failed download fails the build and
# leaves nothing behind.
#
# Placed before the source is copied, so code changes reuse this layer rather
# than downloading again. The pack name must match MODEL_PACK in config.py.
ENV PRESENCE_MODEL_DIR=/opt/models
RUN python -c "from insightface.utils.storage import ensure_available; ensure_available('models', 'buffalo_l', root='/opt/models')" \
 && rm -f /opt/models/models/buffalo_l.zip \
 && test -s /opt/models/models/buffalo_l/w600k_r50.onnx

COPY config.py ./
COPY engine ./engine
COPY server ./server
COPY web ./web

# The database lives under /data, so mount a volume there. Without one the
# kiosk still runs, but forgets every enrolment on restart.
#
# Runs as root because hosting platforms mount volumes root-owned; a non-root
# user would be unable to write the database.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PRESENCE_DATA_DIR=/data

EXPOSE 8000
CMD ["python", "-m", "server.app"]
