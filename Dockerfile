# Presence kiosk server.
#
# Model weights are deliberately not baked in. At ~300MB they would bloat every
# image push, so they download on first boot into PRESENCE_MODEL_DIR. Point
# that at a persistent volume and the download happens once, not every deploy.
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

COPY config.py ./
COPY engine ./engine
COPY server ./server
COPY web ./web

# Everything that must outlive a container lives under /data, so mount a
# volume there. Without one the kiosk still runs, but forgets every enrolment
# and re-downloads the models on each restart.
#
# Runs as root because hosting platforms mount volumes root-owned; a non-root
# user would be unable to write the database.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PRESENCE_DATA_DIR=/data \
    PRESENCE_MODEL_DIR=/data/models

EXPOSE 8000
CMD ["python", "-m", "server.app"]
