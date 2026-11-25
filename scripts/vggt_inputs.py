"""Selection, portable exports and preview helpers. No PyTorch or VGGT imports."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np
from PIL import Image

SCHEMA_VERSION = 1
ARRAY_KEYS = ("frames_rgb", "points_normalized", "occluded", "point_ids", "frame_indices")


def _positive_int(value, name, minimum=1, allow_none=False):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}" + (" or None." if allow_none else "."))


def validate_sample(sample):
    video = np.asarray(sample["video"])
    points = np.asarray(sample["points"])
    occ = np.asarray(sample["occluded"])
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8 or min(video.shape[:3]) < 1:
        raise ValueError("Expected nonempty RGB uint8 video [T,H,W,3].")
    if points.ndim != 3 or points.shape[1:] != (len(video), 2) or occ.shape != points.shape[:2]:
        raise ValueError("Expected normalized points [N,T,2] and occluded [N,T].")
    if points.dtype.kind != "f" or occ.dtype != np.bool_:
        raise ValueError("Expected floating-point annotations and Boolean occlusion flags.")
    return video, points, occ


def sequence_table(data):
    rows = []
    for name in sorted(data):
        video, points, _ = validate_sample(data[name])
        t, h, w, _ = video.shape
        rows.append(f"<tr><td>{html.escape(str(name))}</td><td>{t}</td><td>{w} x {h}</td><td>{len(points)}</td></tr>")
    return "<table><tr><th>Sequence</th><th>Frames</th><th>Resolution (W x H)</th><th>Annotated points</th></tr>" + "".join(rows) + "</table>"


def normalize_spec(spec):
    defaults = dict(start_frame=0, frame_stride=1, max_frames=2, point_ids=None, max_points=None, notes="")
    unknown = set(spec) - set(defaults) - {"name", "sequence"}
    if unknown:
        raise ValueError(f"Unknown selection settings: {sorted(unknown)}")
    result = {**defaults, **spec}
    if not isinstance(result.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", result["name"]):
        raise ValueError("Experiment name: 1–80 letters/digits/underscores/hyphens, starting with a letter or digit.")
    if not isinstance(result.get("sequence"), str) or not result["sequence"]:
        raise ValueError("Select a sequence name.")
    _positive_int(result["start_frame"], "start_frame", 0)
    _positive_int(result["frame_stride"], "frame_stride")
    _positive_int(result["max_frames"], "max_frames", 2, True)
    _positive_int(result["max_points"], "max_points", 1, True)
    if not isinstance(result["notes"], str):
        raise ValueError("notes must be text.")
    ids = result["point_ids"]
    if ids is not None:
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("point_ids must be None or a nonempty list of original DAVIS point IDs.")
        for value in ids:
            _positive_int(value, "point ID", 0)
        if len(set(ids)) != len(ids):
            raise ValueError("Point IDs must be unique.")
        result["point_ids"] = [int(x) for x in ids]
    for key in ("start_frame", "frame_stride", "max_frames", "max_points"):
        if result[key] is not None:
            result[key] = int(result[key])
    return result


def select_clip(data, spec, source_path):
    spec = normalize_spec(spec)
    if spec["sequence"] not in data:
        raise KeyError(f"Unknown DAVIS sequence: {spec['sequence']}")
    video, points, occ = validate_sample(data[spec["sequence"]])
    indices = np.arange(spec["start_frame"], len(video), spec["frame_stride"], dtype=np.int64)[:spec["max_frames"]]
    if len(indices) < 2:
        raise ValueError("Fewer than two frames selected; change the start/stride/frame limit.")
    ids = np.arange(len(points), dtype=np.int64) if spec["point_ids"] is None else np.asarray(spec["point_ids"], dtype=np.int64)
    if not len(ids) or np.any(ids >= len(points)):
        raise ValueError("No point IDs selected or a point ID is outside this sequence's annotation range.")
    # Advanced indexing copies only the selection; exports do not retain the full video/dataset.
    clip = dict(
        frames_rgb=video[indices].copy(),
        points_normalized=points[ids][:, indices].astype(np.float32, copy=True),
        occluded=occ[ids][:, indices].copy(),
        point_ids=ids.copy(),
        frame_indices=indices.copy(),
        metadata=dict(schema_version=SCHEMA_VERSION, dataset="TAP-Vid DAVIS",
                      source_path=str(Path(source_path).expanduser().resolve()),
                      created_utc=datetime.now(timezone.utc).isoformat(), selection=spec),
    )
    validate_clip(clip)
    return clip


def validate_clip(clip):
    metadata = clip.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected export schema_version={SCHEMA_VERSION}.")
    normalize_spec(metadata["selection"])
    frames, points, occ = validate_sample(dict(video=clip["frames_rgb"], points=clip["points_normalized"], occluded=clip["occluded"]))
    ids, indices = clip["point_ids"], clip["frame_indices"]
    if len(frames) < 2 or len(points) < 1:
        raise ValueError("An experiment needs at least two frames and one annotated point.")
    for name, array, length in [("point_ids", ids, len(points)), ("frame_indices", indices, len(frames))]:
        if array.shape != (length,) or array.dtype.kind not in "iu" or np.any(array < 0) or len(np.unique(array)) != length:
            raise ValueError(f"Invalid {name}.")
    if not np.all(np.diff(indices) > 0):
        raise ValueError("Frame indices must be strictly increasing.")


def save_arrays(path, **arrays):
    """Never overwrite existing files; remove only our own incomplete file on failure."""
    path = Path(path)
    with path.open("xb") as stream:
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            stream.close()
            path.unlink(missing_ok=True)
            raise


def save_clip(path, clip):
    validate_clip(clip)
    save_arrays(path, **{key: clip[key] for key in ARRAY_KEYS},
                metadata_json=np.asarray(json.dumps(clip["metadata"], allow_nan=False)))


def load_clip(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_KEYS) | {"metadata_json"}:
            raise ValueError("Not a VGGT selection export (unexpected/missing fields).")
        metadata = archive["metadata_json"]
        if metadata.ndim != 0 or metadata.dtype.kind not in "US":
            raise ValueError("Expected scalar JSON metadata, not pickled Python objects.")
        clip = {key: archive[key] for key in ARRAY_KEYS}
        clip["metadata"] = json.loads(metadata.item())
    validate_clip(clip)
    return clip


def export_batch(data, specs, source_path, destination):
    """Write selections sequentially, never retaining an entire batch of copied frames."""
    specs = [normalize_spec(spec) for spec in specs]
    if not specs or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Queue at least one experiment with unique names.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=SCHEMA_VERSION, experiments=[])
    for spec in specs:
        clip = select_clip(data, spec, source_path)
        path = destination / (spec["name"] + ".npz")
        save_clip(path, clip)
        manifest["experiments"].append(dict(file=path.name, selection=spec,
                                            frames=clip["frame_indices"].tolist(), point_ids=clip["point_ids"].tolist()))
        print(f"Exported {path.name}: {len(clip['frames_rgb'])} frames, {len(clip['point_ids'])} annotated points", flush=True)
        del clip
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def export_iterations(data, specs, source_path, destination):
    """Append numbered experiments under each selected frame window; return new paths."""
    specs = [normalize_spec(spec) for spec in specs]
    if not specs or len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("Queue at least one experiment with unique names.")

    paths = []
    for spec in specs:
        clip = select_clip(data, spec, source_path)
        frames = clip["frame_indices"].tolist()
        sequence = spec["sequence"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sequence):
            raise ValueError(f"Unsupported sequence name for an export folder: {sequence!r}")
        stride = spec["frame_stride"]
        suffix = f"_stride{stride}" if stride != 1 else ""
        folder = Path(destination) / f"{sequence}_{frames[0]}_{frames[-1] + 1}{suffix}"
        folder.mkdir(parents=True, exist_ok=True)

        manifest_path = folder / "manifest.json"
        manifest = dict(schema_version=SCHEMA_VERSION, experiments=[])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION
                    or not isinstance(manifest.get("experiments"), list)):
                raise ValueError(f"Unexpected manifest format: {manifest_path}")

        numbers = [int(match[1]) for entry in folder.iterdir()
                   if (match := re.fullmatch(r"experiment_([0-9]+)\.npz", entry.name))]
        name = f"experiment_{max(numbers, default=0) + 1:03d}"
        path = folder / f"{name}.npz"
        selection = {**spec, "name": name}
        clip["metadata"].update(selection=selection, queued_name=spec["name"])
        save_clip(path, clip)  # Exclusive creation: never overwrite an existing export.
        paths.append(path)
        manifest["experiments"].append(dict(file=path.name, selection=selection, queued_name=spec["name"],
                                            frames=frames, point_ids=clip["point_ids"].tolist()))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(manifest, stream, indent=2)
                stream.close()
                temporary.replace(manifest_path)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Exported {folder.name}/{path.name}: {len(frames)} frames, {len(clip['point_ids'])} annotated points", flush=True)
        del clip
    return paths


def prepare_model_inputs(clip, preprocess):
    """Call VGGT's native preprocessing and map annotations through its resize/crop."""
    validate_clip(clip)
    frames = clip["frames_rgb"]
    h, w = frames.shape[1:3]
    resized_w = 518
    resized_h = round(h * (resized_w / w) / 14) * 14
    if resized_h < 14:
        raise ValueError("Image aspect ratio is too extreme for native VGGT preprocessing.")
    crop_top = max(0, (resized_h - 518) // 2)
    output_h = min(resized_h, 518)
    with tempfile.TemporaryDirectory(prefix="vggt_selected_") as tmp:
        paths = []
        for i, frame in enumerate(frames):
            path = Path(tmp) / f"{i:06}.png"
            Image.fromarray(frame).save(path)
            paths.append(str(path))
        images = preprocess(paths, mode="crop")
    if tuple(images.shape) != (len(frames), 3, output_h, resized_w):
        raise RuntimeError("VGGT preprocessing differs from the expected crop transform; do not use mismatched annotation coordinates.")
    points = clip["points_normalized"] * np.array([resized_w, resized_h], dtype=np.float32) - np.array([0, crop_top], dtype=np.float32)
    in_crop = np.isfinite(points).all(axis=-1)
    in_crop &= (points[..., 0] >= 0) & (points[..., 0] < resized_w) & (points[..., 1] >= 0) & (points[..., 1] < output_h)
    eligible = np.flatnonzero((~clip["occluded"][:, 0]) & in_crop[:, 0])
    eligible = eligible[:clip["metadata"]["selection"]["max_points"]]
    if not len(eligible):
        raise ValueError("No selected points are visible and in-crop in the first input frame. Choose other points/start frame.")
    annotations = dict(point_ids=clip["point_ids"][eligible], frame_indices=clip["frame_indices"].copy(),
                       points_normalized=clip["points_normalized"][eligible], points_input_xy=points[eligible],
                       occluded=clip["occluded"][eligible], in_crop=in_crop[eligible],
                       query_points_xy=points[eligible, 0].copy())
    transform = dict(source_hw=[h, w], resized_hw=[resized_h, resized_w], input_hw=[output_h, resized_w],
                     crop_top=crop_top, mode="crop")
    return images, annotations, transform


def make_video_preview(sample, fps=10.0, max_width=640, show_annotations=False):
    frames, points, occ = validate_sample(sample)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Preview FPS must be finite and positive.")
    _positive_int(max_width, "Preview width")
    source_h, source_w = frames.shape[1:3]
    width = min(source_w, max_width)
    height = max(1, round(source_h * width / source_w))
    encoded_frames, visible_points = [], []
    for t, frame in enumerate(frames):
        image = Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=80)
            encoded_frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        xy = points[:, t]
        visible = (~occ[:, t]) & np.isfinite(xy).all(axis=-1)
        visible &= (xy >= 0).all(axis=-1) & (xy < 1).all(axis=-1)
        visible_points.append([[int(i), float(xy[i, 0] * width), float(xy[i, 1] * height)] for i in np.flatnonzero(visible)])
    payload = dict(frames=encoded_frames, points=visible_points, width=width, height=height,
                   fps=float(fps), show=bool(show_annotations))
    template = '''
<div id="__PLAYER_ID__" style="max-width:100%;padding:8px;border:1px solid #aaa">
  <canvas style="display:block;max-width:100%;height:auto"></canvas>
  <button class="play" type="button">Play</button>
  <input class="slider" type="range" min="0" step="1" value="0" aria-label="Frame index" style="width:45%">
  <span class="counter"></span><br>
  <label>Viewing FPS <input class="fps" type="number" min="0.1" step="0.5" style="width:80px"></label>
  <label><input class="annotations" type="checkbox"> Visible ground-truth points and IDs</label>
</div>
<script>
(() => {
  const root = document.getElementById('__PLAYER_ID__');
  const data = __PLAYER_DATA__;
  const canvas = root.querySelector('canvas'), ctx = canvas.getContext('2d');
  const play = root.querySelector('.play'), slider = root.querySelector('.slider');
  const counter = root.querySelector('.counter'), fps = root.querySelector('.fps');
  const annotations = root.querySelector('.annotations');
  canvas.width = data.width; canvas.height = data.height;
  slider.max = data.frames.length - 1; fps.value = data.fps; annotations.checked = data.show;
  let timer = null, ticket = 0;
  function pause() { clearInterval(timer); timer = null; play.textContent = 'Play'; }
  function render(index) {
    slider.value = index; counter.textContent = 'Frame ' + index + ' / ' + slider.max;
    const current = ++ticket, image = new Image();
    image.onload = () => {
      if (current !== ticket) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0);
      if (annotations.checked) {
        ctx.font = '12px sans-serif'; ctx.lineWidth = 3; ctx.strokeStyle = 'black'; ctx.fillStyle = 'lime';
        for (const [id, x, y] of data.points[index]) {
          ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.stroke(); ctx.fill();
          ctx.strokeText(String(id), x + 5, y - 5); ctx.fillText(String(id), x + 5, y - 5);
        }
      }
    };
    image.src = 'data:image/jpeg;base64,' + data.frames[index];
  }
  function start() {
    const rate = Number(fps.value);
    if (!Number.isFinite(rate) || rate <= 0) return;
    if (Number(slider.value) >= data.frames.length - 1) render(0);
    play.textContent = 'Pause';
    timer = setInterval(() => {
      if (!root.isConnected || Number(slider.value) >= data.frames.length - 1) { pause(); return; }
      render(Number(slider.value) + 1);
    }, 1000 / rate);
  }
  play.onclick = () => { if (timer === null) start(); else pause(); };
  slider.oninput = () => { pause(); render(Number(slider.value)); };
  annotations.onchange = () => render(Number(slider.value));
  fps.onchange = () => { if (timer !== null) { pause(); start(); } };
  render(0);
})();
</script>
'''
    return template.replace("__PLAYER_ID__", "davis_" + uuid.uuid4().hex).replace("__PLAYER_DATA__", json.dumps(payload, allow_nan=False))
