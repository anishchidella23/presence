/* Drawing the engine's verdicts onto a canvas.
 *
 * Every colour and label here comes from the state the engine returned. The
 * browser makes no decisions about what a face is - it only renders them.
 */

const STATE_COLOURS = {
  too_blurry: "#f5a451",
  too_small: "#f5a451",
  unknown: "#ef5f5f",
  confirming: "#4fc3e8",
  recognised: "#f5c451",
  challenged: "#f5c451",
  verified: "#3ddc97",
  spoof_suspected: "#ef5f5f",
};

const STATE_LABELS = {
  too_blurry: "Too blurry",
  too_small: "Move closer",
  unknown: "Unknown",
  confirming: "Confirming",
  recognised: "Recognised",
  challenged: "Verifying",
  verified: "Verified",
  spoof_suspected: "Failed",
};

export function drawFaces(canvas, result) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!result || !result.faces) return;

  // The engine works in the coordinates of the frame it was sent, which is
  // downscaled from what the canvas displays.
  const sx = canvas.width / result.frame_width;
  const sy = canvas.height / result.frame_height;

  for (const face of result.faces) {
    const [x1, y1, x2, y2] = face.bbox;
    const x = x1 * sx, y = y1 * sy;
    const w = (x2 - x1) * sx, h = (y2 - y1) * sy;
    const colour = STATE_COLOURS[face.state] || "#8b9bb0";

    ctx.lineWidth = face.state === "verified" ? 3 : 2;
    ctx.strokeStyle = colour;
    ctx.strokeRect(x, y, w, h);

    // Text is drawn un-mirrored inside a mirrored canvas, otherwise every
    // label would read backwards.
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    const mx = canvas.width - x - w;

    const top = `[${face.track_id}] ${STATE_LABELS[face.state] || face.state}`;
    ctx.font = "600 13px ui-monospace, SFMono-Regular, monospace";
    const topWidth = ctx.measureText(top).width + 12;
    ctx.fillStyle = colour;
    ctx.fillRect(mx, Math.max(0, y - 20), topWidth, 20);
    ctx.fillStyle = "#0b0f14";
    ctx.fillText(top, mx + 6, Math.max(0, y - 20) + 14);

    const bottom = face.name
      ? `${face.name} (${face.confidence.toFixed(0)}%)`
      : face.detail || "";
    if (bottom) {
      ctx.font = "600 14px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = colour;
      ctx.fillRect(mx, y + h, Math.max(w, ctx.measureText(bottom).width + 12), 22);
      ctx.fillStyle = "#0b0f14";
      ctx.fillText(bottom, mx + 6, y + h + 16);
    }
    ctx.restore();
  }
}

/* The prompt belongs to whichever face is actively being challenged. With
 * several people in frame the largest is the one standing at the kiosk. */
export function activeFace(result) {
  if (!result || !result.faces.length) return null;
  const area = (f) => (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
  return result.faces.reduce((a, b) => (area(b) > area(a) ? b : a));
}
