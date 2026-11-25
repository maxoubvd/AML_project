"""Read-only analysis of saved VGGT runs. No torch, model or DAVIS pickle imports.

Tracking metrics are exploratory diagnostics for first-frame-visible queries,
not official TAP-Vid scores. Analysis files are written only by save_analysis().
"""
from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image


def scan_runs(root):
    rows = []
    for directory in sorted(Path(root).glob("*/*")):
        if not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        error_path = directory / "error.json"
        if not metadata_path.exists() and not error_path.exists() and not (directory / "run.log").exists():
            continue
        row = dict(path=str(directory.resolve()), run=directory.parent.name, experiment=directory.name,
                   status="incomplete", sequence="", frames=None, queries=None, device="", seconds=None, input_id="", notes="")
        try:
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                selection = metadata.get("export_metadata", {}).get("selection", metadata.get("selection", {}))
                complete = metadata.get("status", "completed") == "completed" and all((directory / f).is_file() for f in ("predictions.npz", "annotations.npz"))
                row.update(status="completed" if complete else "incomplete", sequence=selection.get("sequence", ""),
                           frames=metadata["input_shape"][0], queries=len(metadata.get("point_ids", [])),
                           device=metadata.get("device", metadata.get("gpu", "")), seconds=metadata.get("inference_and_cpu_copy_seconds"),
                           input_id=metadata.get("export_sha256", "")[:12], notes=selection.get("notes", ""))
            elif error_path.exists():
                error = json.loads(error_path.read_text())
                row.update(status="failed", notes=error.get("error_type", "") + ": " + error.get("message", "").split("\n")[0][:160])
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="invalid", notes=str(error))
        rows.append(row)
    return rows


def table_html(rows, columns):
    def format_value(value):
        if value is None:
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value)
    head = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{html.escape(format_value(row.get(c)))}</td>" for c in columns) + "</tr>" for row in rows)
    return "<table>" + head + body + "</table>"


def load_run(directory, with_images=True):
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("status", "completed") != "completed":
        raise ValueError("Select a completed run, not a failed or interrupted experiment.")
    with np.load(directory / "annotations.npz", allow_pickle=False) as archive:
        annotations = {key: archive[key] for key in archive.files}
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        track = archive["track"]
        vis = archive["vis"]
        conf = archive["conf"] if "conf" in archive.files else None
        frames = None
        if with_images:
            images = archive["images"]
            if images.ndim != 5 or images.shape[0] != 1 or images.shape[2] != 3:
                raise ValueError("Expected saved images [1,S,3,H,W].")
            frames = np.rint(np.clip(images[0].transpose(0, 2, 3, 1), 0, 1) * 255).astype(np.uint8)
            del images
    if track.ndim != 4 or track.shape[0] != 1 or track.shape[-1] != 2:
        raise ValueError("Expected saved tracks [1,S,N,2].")
    s, n = track.shape[1:3]
    expected = dict(point_ids=(n,), frame_indices=(s,), points_normalized=(n, s, 2),
                    points_input_xy=(n, s, 2), occluded=(n, s), in_crop=(n, s), query_points_xy=(n, 2))
    for key, shape in expected.items():
        if key not in annotations or annotations[key].shape != shape:
            raise ValueError(f"Missing or incompatible annotations: {key}, expected {shape}.")
    if vis.shape != (1, s, n) or (conf is not None and conf.shape != vis.shape):
        raise ValueError("Visibility/confidence shapes do not match tracks.")
    if len(np.unique(annotations["point_ids"])) != n or not np.all(np.diff(annotations["frame_indices"]) > 0):
        raise ValueError("Expected unique point IDs and increasing frame indices.")
    transform = metadata["transform"]
    if frames is not None and (len(frames) != s or list(frames.shape[1:3]) != transform["input_hw"]):
        raise ValueError("Saved image dimensions do not match annotations/transform.")
    # Only tracking arrays/images are loaded here; dense depth and point maps remain on disk.
    return dict(path=directory, metadata=metadata, annotations=annotations,
                track_input=track[0].transpose(1, 0, 2), visibility=vis[0].T,
                confidence=conf[0].T if conf is not None else None, frames_rgb=frames)


def diagnostics(run, visibility_threshold=0.5):
    if not np.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1:
        raise ValueError("Visibility threshold must be between 0 and 1.")
    a, tr = run["annotations"], run["metadata"]["transform"]
    resized_h, resized_w = tr["resized_hw"]
    predicted_normalized = (run["track_input"] + np.array([0, tr["crop_top"]], dtype=np.float32)) / np.array([resized_w, resized_h], dtype=np.float32)
    difference = predicted_normalized - a["points_normalized"]
    epe_256 = np.linalg.norm(difference * 256.0, axis=-1)
    source_h, source_w = tr["source_hw"]
    epe_source = np.linalg.norm(difference * np.array([source_w, source_h]), axis=-1)
    after_query = np.ones(a["occluded"].shape, dtype=bool)
    after_query[:, 0] = False
    finite_gt = np.isfinite(a["points_normalized"]).all(axis=-1)
    position_mask = after_query & (~a["occluded"]) & a["in_crop"] & finite_gt
    invalid_positions = position_mask & ~np.isfinite(epe_256)
    epe_256 = np.where(np.isfinite(epe_256), epe_256, np.inf)
    epe_source = np.where(np.isfinite(epe_source), epe_source, np.inf)
    # GT-visible points outside the input crop cannot be fairly scored for visibility.
    # GT-occluded points remain scoreable, even if their hidden coordinates are undefined.
    visibility_mask = after_query & (a["occluded"] | (a["in_crop"] & finite_gt))
    finite_vis = np.isfinite(run["visibility"])
    predicted_visible = finite_vis & (run["visibility"] > visibility_threshold)
    gt_visible = ~a["occluded"]
    visibility_correct = finite_vis & (predicted_visible == gt_visible)
    return dict(source_path=str(run["path"]), epe_256=epe_256, epe_source=epe_source, position_mask=position_mask,
                invalid_positions=invalid_positions, visibility_mask=visibility_mask,
                visibility_correct=visibility_correct, predicted_visible=predicted_visible,
                gt_visible=gt_visible, invalid_visibility=visibility_mask & ~finite_vis,
                predicted_normalized=predicted_normalized, threshold=float(visibility_threshold))


def _check_diagnostics(run, diag):
    if diag.get("source_path") != str(run["path"]) or diag["position_mask"].shape != run["annotations"]["occluded"].shape:
        raise ValueError("Diagnostics belong to another run; recompute them after changing the selected run.")


def _mean(values):
    return float(np.mean(values)) if values.size else None


def summarize(run, diag, indices=None):
    _check_diagnostics(run, diag)
    if indices is None:
        indices = np.arange(len(run["annotations"]["point_ids"]))
    indices = np.asarray(indices, dtype=int)
    mask = diag["position_mask"][indices]
    values = diag["epe_256"][indices][mask]
    source_values = diag["epe_source"][indices][mask]
    vis_mask = diag["visibility_mask"][indices]
    return dict(points=len(indices), scored_pairs=int(mask.sum()), mean_epe_256=_mean(values),
                median_epe_256=float(np.median(values)) if values.size else None,
                max_epe_256=float(np.max(values)) if values.size else None,
                mean_epe_source_px=_mean(source_values),
                nonfinite_positions=int(diag["invalid_positions"][indices].sum()),
                visibility_pairs=int(vis_mask.sum()),
                visibility_agreement=_mean(diag["visibility_correct"][indices][vis_mask]),
                always_visible_baseline=_mean(diag["gt_visible"][indices][vis_mask]),
                nonfinite_visibility=int(diag["invalid_visibility"][indices].sum()))


def per_point_rows(run, diag):
    rows = []
    for i, point_id in enumerate(run["annotations"]["point_ids"]):
        rows.append(dict(point_id=int(point_id), **summarize(run, diag, [i])))
    return sorted(rows, key=lambda r: -(r["mean_epe_256"] if r["mean_epe_256"] is not None else -1))


def per_frame_rows(run, diag):
    _check_diagnostics(run, diag)
    rows = []
    for t, frame_id in enumerate(run["annotations"]["frame_indices"]):
        mask = diag["position_mask"][:, t]
        vis_mask = diag["visibility_mask"][:, t]
        rows.append(dict(frame_id=int(frame_id), scored_points=int(mask.sum()),
                         mean_epe_256=_mean(diag["epe_256"][:, t][mask]),
                         nonfinite_positions=int(diag["invalid_positions"][:, t].sum()),
                         visibility_agreement=_mean(diag["visibility_correct"][:, t][vis_mask])))
    return rows


def group_rows(run, diag, groups):
    known = {int(point_id): i for i, point_id in enumerate(run["annotations"]["point_ids"])}
    used, rows = set(), []
    for name, ids in groups.items():
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Group {name}: supply a nonempty list of unique original point IDs.")
        unknown = set(ids) - set(known)
        if unknown:
            raise ValueError(f"Group {name} has IDs not tracked in this run: {sorted(unknown)}")
        if used.intersection(ids):
            raise ValueError("Use disjoint point groups for this comparison.")
        used.update(ids)
        rows.append(dict(group=str(name), point_ids=list(ids), **summarize(run, diag, [known[i] for i in ids])))
    return rows


def worst_pairs(run, diag, limit=10):
    _check_diagnostics(run, diag)
    positions = np.argwhere(diag["position_mask"])
    if not len(positions):
        return []
    values = diag["epe_256"][diag["position_mask"]]
    rows = []
    for index in np.argsort(-values, kind="stable")[:limit]:
        i, t = positions[index]
        rows.append(dict(point_id=int(run["annotations"]["point_ids"][i]),
                         frame_id=int(run["annotations"]["frame_indices"][t]),
                         epe_256=float(diag["epe_256"][i, t]),
                         visibility_score=float(run["visibility"][i, t]),
                         confidence_score=float(run["confidence"][i, t]) if run["confidence"] is not None else None))
    return rows


def compare_runs(paths, visibility_threshold=0.5, labels=None):
    """Separate per-run summaries; timing ratios only for verified matched inputs/model."""
    rows, reference = [], None
    for index, path in enumerate(paths):
        run = load_run(path, with_images=False)
        meta = run["metadata"]
        diag = diagnostics(run, visibility_threshold)
        notes = []
        matched = False
        if reference is None:
            reference = run
            notes.append("Reference run")
        else:
            ref = reference["metadata"]
            same_export = bool(meta.get("export_sha256")) and meta.get("export_sha256") == ref.get("export_sha256")
            same_annotations = all(np.array_equal(run["annotations"][key], reference["annotations"][key], equal_nan=True)
                                   for key in ("frame_indices", "point_ids", "points_normalized", "occluded", "in_crop"))
            same_model = all(meta.get(key) not in (None, "unknown") and meta.get(key) == ref.get(key)
                             for key in ("vggt_git_revision", "checkpoint_path", "checkpoint_bytes", "checkpoint_mtime_ns"))
            matched = same_export and same_annotations and same_model and meta["transform"] == ref["transform"]
            if not matched:
                notes.append("Unmatched inputs/model: do not interpret timing ratio or error difference as a controlled comparison")
            else:
                notes.append("Same export, annotations, preprocessing and recorded model/checkpoint")
            if meta.get("autocast_dtype") != ref.get("autocast_dtype"):
                notes.append("Different inference precision")
        seconds = meta.get("inference_and_cpu_copy_seconds")
        ref_seconds = reference["metadata"].get("inference_and_cpu_copy_seconds")
        ratio = seconds / ref_seconds if matched and seconds is not None and ref_seconds is not None and ref_seconds > 0 else None
        rows.append(dict(label=(labels or {}).get(index, Path(path).parent.name), path=str(Path(path).resolve()),
                         sequence=meta.get("export_metadata", {}).get("selection", {}).get("sequence", ""),
                         frames=len(run["annotations"]["frame_indices"]), device=meta.get("device"),
                         autocast=meta.get("autocast_dtype") or "disabled (FP32)", seconds=seconds,
                         time_ratio_to_first=ratio, comparison_note="; ".join(notes), **summarize(run, diag)))
    return rows


def plot_tracking(run, diag, point_id, frame_id):
    _check_diagnostics(run, diag)
    import matplotlib.pyplot as plt
    a = run["annotations"]
    point_matches = np.flatnonzero(a["point_ids"] == point_id)
    frame_matches = np.flatnonzero(a["frame_indices"] == frame_id)
    if not len(point_matches) or not len(frame_matches):
        raise ValueError("Choose an original point/frame ID present in this run.")
    if run["frames_rgb"] is None:
        raise ValueError("Load this run with images to draw overlays.")
    i, t = int(point_matches[0]), int(frame_matches[0])
    figures = {}
    frame_rows = per_frame_rows(run, diag)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(a["frame_indices"], [r["mean_epe_256"] if r["mean_epe_256"] is not None else np.nan for r in frame_rows], marker="o")
    ax.set(xlabel="Stored video frame index", ylabel="Mean EPE (256-space px)", title="Visible, in-crop points; query frame excluded")
    figures["error_by_frame"] = fig
    rows = per_point_rows(run, diag)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar([str(r["point_id"]) for r in rows], [r["mean_epe_256"] if r["mean_epe_256"] is not None else np.nan for r in rows])
    ax.set(xlabel="Original point ID (ranked by error)", ylabel="Mean EPE (256-space px)")
    figures["error_by_point"] = fig
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(run["frames_rgb"][t])
    valid_gt = (~a["occluded"][:, t]) & a["in_crop"][:, t] & np.isfinite(a["points_input_xy"][:, t]).all(axis=-1)
    gt, pred = a["points_input_xy"][:, t], run["track_input"][:, t]
    finite_pred = np.isfinite(pred).all(axis=-1)
    ax.scatter(*gt[valid_gt].T, c="lime", s=22, label="GT visible / in crop")
    ax.scatter(*pred[finite_pred].T, c="red", marker="x", s=22, label="Prediction (not confidence-filtered)")
    for j in np.flatnonzero(valid_gt & finite_pred):
        ax.plot([gt[j, 0], pred[j, 0]], [gt[j, 1], pred[j, 1]], c="yellow", linewidth=.7)
    for j in np.flatnonzero(valid_gt):
        ax.text(gt[j, 0] + 3, gt[j, 1] - 3, str(a["point_ids"][j]), color="lime", fontsize=8)
    h, w = run["frames_rgb"][t].shape[:2]
    ax.set(xlim=(-.5, w-.5), ylim=(h-.5, -.5), title=f"Stored frame {frame_id}: GT / prediction")
    ax.legend(loc="upper right")
    figures[f"overlay_frame_{frame_id}"] = fig
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    gt_shown = np.where(((~a["occluded"][i]) & a["in_crop"][i])[:, None], a["points_input_xy"][i], np.nan)
    for dimension, axis in enumerate(axes[:2]):
        axis.plot(a["frame_indices"], gt_shown[:, dimension], "g.-", label="Ground truth (visible / in crop)")
        axis.plot(a["frame_indices"], run["track_input"][i, :, dimension], "r.-", label="Prediction")
        axis.set_ylabel("XY"[dimension] + " (input px)")
        axis.legend()
    axes[2].plot(a["frame_indices"], np.where(diag["position_mask"][i], diag["epe_256"][i], np.nan), "o-")
    axes[2].set_ylabel("Scored EPE (256 px)")
    axes[3].step(a["frame_indices"], (~a["occluded"][i]).astype(int), where="mid", label="GT visibility")
    axes[3].plot(a["frame_indices"], run["visibility"][i], label="Predicted visibility score")
    if run["confidence"] is not None:
        axes[3].plot(a["frame_indices"], run["confidence"][i], "--", label="Tracking confidence score")
    axes[3].axhline(diag["threshold"], color="grey", linestyle=":", label="Visibility threshold")
    axes[3].set(xlabel="Stored video frame index", ylabel="Score / visibility")
    axes[3].legend()
    fig.suptitle(f"Original point ID {point_id}; missing GT/error segments are not scored")
    fig.tight_layout()
    figures[f"point_{point_id}_trajectory"] = fig
    return figures


def plot_geometry(run, frame_id, include_points=False, point_stride=8):
    import matplotlib.pyplot as plt
    matches = np.flatnonzero(run["annotations"]["frame_indices"] == frame_id)
    if not len(matches) or run["frames_rgb"] is None or point_stride < 1:
        raise ValueError("Choose a saved frame, load images, and use a positive point stride.")
    t = int(matches[0])
    figures = {}
    with np.load(run["path"] / "predictions.npz", allow_pickle=False) as archive:
        depth = archive["depth"][0, t, ..., 0].copy()
        confidence = archive["depth_conf"][0, t].copy()
        extrinsics = archive["extrinsic"][0].copy()
        # NPZ decompresses each requested array; copy the selected frame to release the rest.
        points = archive["world_points"][0, t].copy() if include_points else None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(run["frames_rgb"][t]); axes[0].set_title(f"Frame {frame_id}")
    image = axes[1].imshow(depth); axes[1].set_title("Depth (relative scale)"); fig.colorbar(image, ax=axes[1])
    image = axes[2].imshow(confidence); axes[2].set_title("Depth confidence (model score)"); fig.colorbar(image, ax=axes[2])
    fig.tight_layout(); figures[f"geometry_frame_{frame_id}"] = fig
    centers = -np.einsum("nji,nj->ni", extrinsics[:, :3, :3], extrinsics[:, :3, 3])
    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    ax.plot(*centers.T, marker="."); ax.set_title("Estimated camera centers — arbitrary scene scale")
    figures["camera_centers"] = fig
    if points is not None:
        points = points[::point_stride, ::point_stride].reshape(-1, 3)
        colors = run["frames_rgb"][t, ::point_stride, ::point_stride].reshape(-1, 3) / 255.
        finite = np.isfinite(points).all(axis=-1)
        fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*points[finite].T, c=colors[finite], s=1)
        ax.set_title("Predicted point map — subsampled, not confidence-filtered")
        figures[f"points_frame_{frame_id}"] = fig
    return figures


def comparison_player(run, threshold=0.5, fps=10.0, width=640):
    frames = run["frames_rgb"]
    if frames is None:
        raise ValueError("Load this run with images to generate a player.")
    if not np.isfinite(fps) or fps <= 0 or not isinstance(width, int) or width < 1 or not 0 <= threshold <= 1:
        raise ValueError("Use positive FPS/width and a threshold between 0 and 1.")
    a = run["annotations"]
    d = diagnostics(run, threshold)
    h, w = frames.shape[1:3]
    out_w, out_h = min(width, w), max(1, round(h * min(width, w) / w))
    scale = np.array([out_w / w, out_h / h])
    encoded, ground_truth, predicted = [], [], []
    for t, frame in enumerate(frames):
        with io.BytesIO() as buffer:
            Image.fromarray(frame).resize((out_w, out_h), Image.Resampling.BILINEAR).save(buffer, format="JPEG", quality=80)
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        gt_rows, pred_rows = [], []
        for i, point_id in enumerate(a["point_ids"]):
            gt, pred = a["points_input_xy"][i, t], run["track_input"][i, t]
            if not a["occluded"][i, t] and a["in_crop"][i, t] and np.isfinite(gt).all():
                gt_rows.append([int(point_id), *map(float, gt * scale)])
            if np.isfinite(pred).all():
                error = d["epe_source"][i, t]
                pred_rows.append([int(point_id), *map(float, pred * scale), bool(d["predicted_visible"][i, t]),
                                  float(error) if d["position_mask"][i, t] and np.isfinite(error) else None])
        ground_truth.append(gt_rows); predicted.append(pred_rows)
    payload = dict(frames=encoded, gt=ground_truth, pred=predicted, ids=a["point_ids"].tolist(),
                   frame_ids=a["frame_indices"].tolist(), width=out_w, height=out_h, fps=float(fps),
                   threshold=float(threshold), above_threshold=d["predicted_visible"].sum(axis=0).tolist())
    template = '''
<div id="__ID__" style="max-width:100%;padding:8px;border:1px solid #aaa">
 <p><span style="color:green">Green: visible ground truth</span> · <span style="color:red">Red: prediction</span> · Yellow: displacement</p>
 <canvas style="display:block;max-width:100%;height:auto"></canvas>
 <button class="play" type="button">Play</button>
 <input class="slider" type="range" min="0" step="1" value="0" style="width:45%" aria-label="Frame position"><span class="counter"></span><br>
 <label>Point ID <select class="point"><option value="all">All tracked points</option></select></label>
 <label>Viewing FPS <input class="fps" type="number" min="0.1" step="0.5" style="width:70px"></label>
 <label><input class="low" type="checkbox" checked> Show low-visibility predictions</label>
 <label><input class="labels" type="checkbox" checked> Point IDs</label>
 <label><input class="errors" type="checkbox" checked> Position error (original video pixels)</label>
 <p class="status" aria-live="off"></p>
</div>
<script>
(() => {
 const root = document.getElementById('__ID__'), data = __DATA__;
 const canvas = root.querySelector('canvas'), ctx = canvas.getContext('2d');
 const play = root.querySelector('.play'), slider = root.querySelector('.slider'), counter = root.querySelector('.counter');
 const point = root.querySelector('.point'), fps = root.querySelector('.fps'), low = root.querySelector('.low');
 const labels = root.querySelector('.labels'), errors = root.querySelector('.errors'), status = root.querySelector('.status');
 canvas.width = data.width; canvas.height = data.height; slider.max = data.frames.length - 1; fps.value = data.fps;
 for (const id of data.ids) { const option = document.createElement('option'); option.value = String(id); option.textContent = String(id); point.appendChild(option); }
 let timer = null, ticket = 0;
 function pause() { clearInterval(timer); timer = null; play.textContent = 'Play'; }
 function wanted(id) { return point.value === 'all' || Number(point.value) === id; }
 function marker(text, x, y, color, cross) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  if (cross) { ctx.moveTo(x-4,y-4);ctx.lineTo(x+4,y+4);ctx.moveTo(x+4,y-4);ctx.lineTo(x-4,y+4);ctx.stroke(); }
  else { ctx.arc(x,y,3,0,2*Math.PI);ctx.fill(); }
  if (text) {
   ctx.font='12px sans-serif';ctx.lineWidth=3;ctx.strokeStyle='black';
   const tx=Math.max(2,Math.min(x+5,canvas.width-ctx.measureText(text).width-2)), ty=Math.max(12,y-5);
   ctx.strokeText(text,tx,ty);ctx.fillText(text,tx,ty);
  }
 }
 function render(t) {
  slider.value=t;counter.textContent='Stored frame '+data.frame_ids[t]+' ('+(t+1)+'/'+data.frames.length+')';
  status.textContent=data.above_threshold[t]+' / '+data.ids.length+' points predicted visible (score > '+data.threshold+').'
   +(t===0?' Query frame: position errors not scored.':' Error labels require visible, in-crop ground truth.');
  const current=++ticket, image=new Image();
  image.onload=()=>{
   if(current!==ticket)return;
   ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);
   const truth=new Map(data.gt[t].map(p=>[p[0],p])), drawnErrors=new Map();
   for(const [id,x,y,visible,error] of data.pred[t]) {
    if(!wanted(id)||(!visible&&!low.checked))continue;
    if(truth.has(id)){const g=truth.get(id);ctx.strokeStyle='yellow';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(g[1],g[2]);ctx.lineTo(x,y);ctx.stroke();}
    if(error!==null)drawnErrors.set(id,error);
    marker(labels.checked&&!truth.has(id)?String(id):'',x,y,'red',true);
   }
   for(const [id,x,y] of data.gt[t])if(wanted(id)){
    const text=[labels.checked?'#'+id:'',errors.checked&&drawnErrors.has(id)?drawnErrors.get(id).toFixed(1)+' px':''].filter(Boolean).join(' · ');
    marker(text,x,y,'lime',false);
   }
  };
  image.src='data:image/jpeg;base64,'+data.frames[t];
 }
 function start(){const rate=Number(fps.value);if(!Number.isFinite(rate)||rate<=0)return;
  if(Number(slider.value)>=data.frames.length-1)render(0);play.textContent='Pause';
  timer=setInterval(()=>{if(!root.isConnected||Number(slider.value)>=data.frames.length-1){pause();return;}render(Number(slider.value)+1);},1000/rate);}
 play.onclick=()=>{if(timer===null)start();else pause();};
 slider.oninput=()=>{pause();render(Number(slider.value));};
 for(const control of [point,low,labels,errors])control.onchange=()=>render(Number(slider.value));
 fps.onchange=()=>{if(timer!==null){pause();start();}};
 render(0);
})();
</script>
'''
    return template.replace("__ID__", "tracking_" + uuid.uuid4().hex).replace("__DATA__", json.dumps(payload, allow_nan=False))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else (None if np.isnan(value) else "Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    return value


def save_analysis(root, run, diag, figures=None, point_groups=None, comparison_rows=None, notes="", *, fps=10.0, width=640):
    """Save a new bundle for one run, making standard plots when none were supplied."""
    _check_diagnostics(run, diag)
    import matplotlib.pyplot as plt

    source = Path(run["path"])
    metadata = run["metadata"]
    selection = metadata.get("export_metadata", {}).get("selection", metadata.get("selection", {}))
    sequence = str(selection.get("sequence") or "unknown-sequence")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", sequence).strip("_") or "unknown-sequence"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    directory = Path(root) / slug / source.parent.name / source.name / stamp
    grouped = group_rows(run, diag, point_groups or {})  # Validate before writing anything.
    worst = worst_pairs(run, diag)
    player = comparison_player(run, threshold=diag["threshold"], fps=fps, width=width)
    title = html.escape(f"{sequence} — {source.parent.name} / {source.name} ({metadata.get('device', 'unknown device')})")
    figures = dict(figures or {})
    generated = {}
    try:
        if "error_by_frame" not in figures:
            a = run["annotations"]
            point_id = worst[0]["point_id"] if worst else int(a["point_ids"][0])
            frame_id = worst[0]["frame_id"] if worst else int(a["frame_indices"][-1])
            with plt.ioff():
                generated = plot_tracking(run, diag, point_id=point_id, frame_id=frame_id)
            figures = {**generated, **figures}
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "tracking.html").write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title></head>'
            f'<body><h1>{title}</h1>{player}</body></html>', encoding="utf-8")
        tables = dict(per_point=per_point_rows(run, diag), per_frame=per_frame_rows(run, diag),
                      worst_pairs=worst, point_groups=grouped, comparison=comparison_rows or [])
        for name, rows in tables.items():
            if rows:
                with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
        for name, figure in figures.items():
            figure.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
        settings = dict(source_run=str(source), source_metadata=metadata, visibility_threshold=diag["threshold"],
                        point_groups=point_groups or {}, notes=notes,
                        position_metric="Euclidean error after mapping coordinates to canonical 256x256; query frame, GT-occluded and out-of-crop points excluded; nonfinite predictions penalized with infinity",
                        visibility_metric="Agreement at configured threshold; query frame and GT-visible/out-of-crop positions excluded; nonfinite scores count as incorrect",
                        protocol="Exploratory, not official TAP-Vid benchmark evaluation", saved_figures=list(figures),
                        player=dict(file="tracking.html", error_unit="original video pixels", viewing_fps=float(fps)),
                        summary=summarize(run, diag))
        (directory / "summary.json").write_text(json.dumps(_json_safe(settings), indent=2, allow_nan=False), encoding="utf-8")
    finally:
        for figure in generated.values():
            plt.close(figure)
    return directory
