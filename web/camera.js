/* Camera access and frame capture, shared by the kiosk and enrolment pages. */

export async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

/* Capture the current video frame as a JPEG data URL, downscaled.
 *
 * Recognition works from a face of roughly a hundred pixels, so sending full
 * resolution spends bandwidth and inference time on detail the pipeline
 * discards anyway. */
export function captureFrame(video, maxEdge = 720, quality = 0.7) {
  const scale = Math.min(1, maxEdge / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", quality);
}

export function fitCanvasToVideo(canvas, video) {
  const rect = video.getBoundingClientRect();
  if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
    canvas.width = Math.round(rect.width);
    canvas.height = Math.round(rect.height);
  }
}
