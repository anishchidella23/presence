"""FastAPI kiosk: frames in over a websocket, verdicts out as JSON.

The browser owns the camera and all rendering; the server owns the pipeline.
That split exists because a Python process holding a camera on macOS is a
permissions fight, because it lets the kiosk run on a phone or tablet pointed
at the same host, and because it keeps every drawing decision out of the
engine.

Pacing is request-response: the browser sends one frame and waits for its
verdict before sending the next. This self-throttles to whatever the server can
actually keep up with, so a slow frame delays the next capture instead of
building a queue of stale frames that are already wrong by the time they are
processed.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import MIN_ENROLMENT_IMAGES, STREAM_MAX_EDGE
from engine.detector import FaceDetector
from engine.gallery import Gallery
from engine.pipeline import PresencePipeline
from engine.store import Store

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent.parent / "web"


class Kiosk:
    """Long-lived objects shared by every request.

    Only the stateless, expensive pieces live here: the detector, which loads
    several hundred megabytes of model, and the gallery. Tracking and liveness
    state belong to a single visitor, so each connection gets its own pipeline
    (see `new_pipeline`) rather than sharing one that every new arrival would
    reset out from under whoever was already mid-challenge.
    """

    def __init__(self) -> None:
        self.store = Store()
        self.detector = FaceDetector()
        self.gallery = Gallery.from_store(self.store)
        self.last_logged: dict[str, bool] = {}
        # Pipelines run in worker threads and share one SQLite connection.
        self._store_lock = threading.Lock()

    def new_pipeline(self) -> PresencePipeline:
        pipeline = PresencePipeline(detector=self.detector, gallery=self.gallery)
        pipeline.on_verified = self._on_verified
        return pipeline

    def _on_verified(self, name: str, result) -> None:
        """Called by a pipeline the moment someone passes liveness."""
        with self._store_lock:
            written = self.store.log_presence(
                name=name,
                confidence=result.confidence,
                similarity=result.similarity,
                challenges_passed=result.challenges_passed,
            )
        self.last_logged[name] = written
        log.info("%s verified (%s)", name, "logged" if written else "already logged today")

    def reload_gallery(self) -> None:
        """Rebuild the gallery after enrolment.

        Open connections pick up the new gallery on their next frame, so a
        person enrolled mid-session can check in without anyone reconnecting.
        """
        with self._store_lock:
            self.gallery = Gallery.from_store(self.store)


kiosk: Kiosk | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kiosk
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("loading models")
    kiosk = Kiosk()
    log.info("kiosk ready: %d enrolled", len(kiosk.gallery))
    yield
    kiosk.store.close()


app = FastAPI(title="Presence", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def decode_frame(payload: str) -> np.ndarray | None:
    """Turn a base64 data URL from the browser into a BGR array."""
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    buffer = np.frombuffer(raw, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


# --- pages ---------------------------------------------------------------


@app.get("/")
def kiosk_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/enrol")
def enrol_page() -> FileResponse:
    return FileResponse(WEB_DIR / "enrol.html")


@app.get("/history")
def history_page() -> FileResponse:
    return FileResponse(WEB_DIR / "history.html")


# --- api -----------------------------------------------------------------


@app.get("/api/roster")
def roster() -> JSONResponse:
    return JSONResponse(kiosk.store.roster())


@app.get("/api/events")
def events(limit: int = 50) -> JSONResponse:
    return JSONResponse([vars(e) for e in kiosk.store.recent_events(limit)])


class EnrolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    images: list[str] = Field(min_length=1)


@app.post("/api/enrol")
def enrol(request: EnrolRequest) -> JSONResponse:
    """Enrol a person from several captured frames.

    Each frame is embedded independently and stored as its own reference, so
    the gallery holds one vector per pose rather than a single averaged one
    that represents no pose particularly well.
    """
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="A name is required")

    accepted, rejected = 0, []
    for index, payload in enumerate(request.images):
        frame = decode_frame(payload)
        if frame is None:
            rejected.append(f"frame {index + 1}: could not be decoded")
            continue
        embedding = kiosk.detector.embed_crop(frame)
        if embedding is None:
            rejected.append(f"frame {index + 1}: no face found")
            continue
        kiosk.store.add_embedding(name, embedding, source=f"enrolment {index + 1}")
        accepted += 1

    if accepted < MIN_ENROLMENT_IMAGES:
        # Roll back rather than leave a half-enrolled person who will match
        # poorly and be hard to diagnose later.
        if accepted:
            kiosk.store.delete_person(name)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {accepted} of {len(request.images)} frames held a usable face; "
                f"{MIN_ENROLMENT_IMAGES} are required. " + "; ".join(rejected[:3])
            ),
        )

    kiosk.reload_gallery()
    return JSONResponse({"name": name, "references": accepted, "rejected": rejected})


@app.delete("/api/people/{name}")
def remove_person(name: str) -> JSONResponse:
    kiosk.store.delete_person(name)
    kiosk.reload_gallery()
    return JSONResponse({"removed": name})


@app.get("/api/config")
def client_config() -> JSONResponse:
    return JSONResponse({"stream_max_edge": STREAM_MAX_EDGE})


# --- live verification ---------------------------------------------------


@app.websocket("/ws/verify")
async def verify(socket: WebSocket) -> None:
    await socket.accept()
    # A pipeline per connection: tracks and liveness challenges belong to the
    # person at this kiosk, and must neither leak to nor be reset by anyone
    # else who connects.
    pipeline = kiosk.new_pipeline()

    try:
        while True:
            frame = decode_frame(await socket.receive_text())
            if frame is None:
                await socket.send_json({"error": "undecodable frame"})
                continue

            pipeline.gallery = kiosk.gallery
            # Inference is CPU-bound and takes a sizeable fraction of a second.
            # Run inline, it would stall the event loop and every other
            # connection with it.
            result = await asyncio.to_thread(pipeline.process, frame)
            payload = result.to_dict()
            payload["logged"] = {
                face.name: kiosk.last_logged.get(face.name, False)
                for face in result.faces
                if face.name
            }
            await socket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("verification socket failed")
        await socket.close(code=1011)


def serve() -> None:
    import uvicorn

    from config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    serve()
