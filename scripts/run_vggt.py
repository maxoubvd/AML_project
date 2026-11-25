#!/usr/bin/env python3
"""Offline VGGT runner: selected exports only, one fresh process per experiment.

Example (run from a terminal AFTER shutting down the selection notebook kernel):
    python scripts/run_vggt.py --input-dir data/vggt_inputs/BATCH --device cuda
    python scripts/run_vggt.py --input-dir data/vggt_inputs/BATCH --validate-only

No package/weight downloads, automatic device fallback, or benchmark analysis.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def git_revision(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_device(requested):
    import torch
    if requested == "cpu":
        return torch.device("cpu"), None
    if requested != "cuda":
        raise ValueError("Device must be cpu or cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Choose --device cpu explicitly or use a CUDA-enabled environment.")
    return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_memory_efficient(model_class, checkpoint, device):
    """No full random CPU model and no second deserialized anonymous weight copy.

    VGGT's DINO constructor reads a tiny linspace schedule with .item(). Keep
    that schedule on CPU while parameter allocations use meta. This scoped,
    thread-local TorchFunctionMode does not patch torch globally or modify VGGT.
    Nonpersistent normalization constants in the local VGGT checkout are real
    CPU buffers; fail clearly if another checkout leaves unresolved meta tensors.
    """
    import torch
    from torch.overrides import TorchFunctionMode

    class ConstructorScalars(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is torch.linspace and "device" not in kwargs:
                kwargs["device"] = "cpu"
            return func(*args, **kwargs)

    with torch.device("meta"), ConstructorScalars():
        model = model_class()
    if not all(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("This model did not support empty parameter initialization; refusing a memory-heavy fallback.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    remaining = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.is_meta]
    if remaining:
        raise RuntimeError(f"Uninitialized meta tensors in this VGGT version: {remaining}")
    return model.to(device=device, dtype=torch.float32).eval()


def predict_arrays(model, images_cpu, query_points_xy, device, amp_dtype):
    import numpy as np
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.inference_mode():
        images = images_cpu.to(device)
        queries = torch.as_tensor(query_points_xy, dtype=torch.float32, device=device)
        context = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()
        with context:
            raw = model(images, query_points=queries)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(raw["pose_enc"].float(), images.shape[-2:])
        raw.update(extrinsic=extrinsic, intrinsic=intrinsic)
        arrays = {}
        for key, value in raw.items():
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().cpu().float().numpy()
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, torch.Tensor) for v in value):
                arrays[key] = np.stack([v.detach().cpu().float().numpy() for v in value])
            else:
                raise TypeError(f"Unexpected model output {key}: {type(value)}")
        return arrays


def run_worker(input_file, output_dir, device_name, vggt_root, checkpoint=None):
    from vggt_inputs import load_clip, prepare_model_inputs, save_arrays

    print(f"[input] Loading selected export: {input_file}", flush=True)
    clip = load_clip(input_file)
    root = Path(vggt_root).expanduser().resolve()
    if not (root / "vggt/models/vggt.py").is_file():
        raise FileNotFoundError(f"VGGT checkout missing at {root}; use --vggt-root PATH.")
    sys.path.insert(0, str(root))
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    device, amp_dtype = configure_device(device_name)
    checkpoint = (Path(checkpoint).expanduser() if checkpoint else Path(torch.hub.get_dir()) / "checkpoints/model.pt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cached checkpoint not found: {checkpoint}. Supply --checkpoint PATH; no automatic download is performed.")
    print("[preprocess] Applying native VGGT crop and mapping annotations", flush=True)
    images, annotations, transform = prepare_model_inputs(clip, load_and_preprocess_images)
    print(f"[preprocess] Shape {tuple(images.shape)}; {len(annotations['point_ids'])} first-frame-visible, in-crop queries", flush=True)
    input_metadata = clip["metadata"]
    del clip  # Raw selected RGB is no longer needed; only preprocessed inputs remain.
    print(f"[load] Empty initialization + memory-mapped checkpoint -> {device}", flush=True)
    model = load_memory_efficient(VGGT, checkpoint, device)
    print("[inference] Model loaded; starting joint inference", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    predictions = predict_arrays(model, images, annotations["query_points_xy"], device, amp_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    metadata = dict(
        status="completed", created_utc=datetime.now(timezone.utc).isoformat(),
        export_path=str(Path(input_file).resolve()), export_sha256=sha256(input_file), export_metadata=input_metadata,
        transform=transform, input_shape=list(images.shape), point_ids=annotations["point_ids"].tolist(),
        query_protocol="exploratory: first selected frame, visible and in-crop; not official TAP-Vid evaluation",
        vggt_checkout=str(root), vggt_git_revision=git_revision(root), project_git_revision=git_revision(PROJECT_ROOT),
        checkpoint_path=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_loader="meta initialization + mmap + assign=True",
        checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
        device=str(next(model.parameters()).device), parameter_dtype=str(next(model.parameters()).dtype),
        autocast_dtype=str(amp_dtype) if amp_dtype is not None else None,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        torch_version=str(torch.__version__), cuda_build=torch.version.cuda,
        inference_and_cpu_copy_seconds=elapsed,
    )
    # All output arrays are now on CPU. Release model/GPU tensors before compression.
    del model, images
    print(f"[save] Writing predictions and annotations to {output_dir}", flush=True)
    output_dir = Path(output_dir)
    save_arrays(output_dir / "predictions.npz", **predictions)
    save_arrays(output_dir / "annotations.npz", **annotations)
    # Metadata is the completion marker: written only after both arrays succeed.
    write_json(output_dir / "metadata.json", metadata)
    print(f"[done] Inference/CPU copy: {elapsed:.2f}s", flush=True)


def discover_inputs(paths, directory):
    files = [Path(p).expanduser().resolve() for p in paths]
    if directory:
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Input directory not found: {directory}")
        files.extend(sorted(directory.glob("*.npz")))
    files = list(dict.fromkeys(files))
    if not files:
        raise ValueError("Provide exported .npz files or --input-dir containing them.")
    for path in files:
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise ValueError(f"Expected a selected .npz export, not the DAVIS pickle: {path}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Input filenames must have unique stems within a batch.")
    return files


def stream_process(command, log_path):
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()


def run_batch(files, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    destination = Path(args.output_root).expanduser().resolve() / stamp
    destination.mkdir(parents=True, exist_ok=False)
    records = [dict(input=str(p), output=p.stem, status="pending") for p in files]
    manifest = dict(status="running", device=args.device, experiments=records)
    manifest_path = destination / "batch.json"
    write_json(manifest_path, manifest)
    print(f"Batch results: {destination}", flush=True)
    failed = False
    for path, record in zip(files, records):
        output = destination / path.stem
        output.mkdir()
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", str(path),
                   "--_output-dir", str(output), "--device", args.device, "--vggt-root", str(args.vggt_root)]
        if args.checkpoint:
            command += ["--checkpoint", str(args.checkpoint)]
        record["status"] = "running"
        write_json(manifest_path, manifest)
        try:
            code = stream_process(command, output / "run.log")
        except BaseException:
            record["status"] = "interrupted"
            manifest["status"] = "interrupted"
            write_json(manifest_path, manifest)
            raise
        record["returncode"] = code
        completed = code == 0 and (output / "metadata.json").is_file()
        record["status"] = "completed" if completed else "failed"
        if not completed:
            failed = True
            print(f"FAILED: {path.name} (return code {code}); inspect {output / 'run.log'}", file=sys.stderr)
            if code < 0:
                print(f"Worker terminated by signal {-code}; check OS logs if memory exhaustion is suspected.", file=sys.stderr)
        write_json(manifest_path, manifest)
        if failed and not args.keep_going:
            break
    manifest["status"] = "failed" if failed else "completed"
    write_json(manifest_path, manifest)
    return 1 if failed else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="Selected .npz exports, never the full DAVIS pickle.")
    parser.add_argument("--input-dir", type=Path, help="Process all .npz exports directly inside this batch folder.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--vggt-root", type=Path, default=PROJECT_ROOT.parent / "vggt")
    parser.add_argument("--checkpoint", type=Path, help="Trusted local .pt checkpoint; default is Torch Hub's cached model.pt.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/vggt")
    parser.add_argument("--validate-only", action="store_true", help="Check exports without importing torch/VGGT or creating results.")
    parser.add_argument("--keep-going", action="store_true", help="Continue other experiments after a failed worker; still return a nonzero exit code.")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_output-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._worker is not None:
        if args._output_dir is None:
            raise ValueError("Internal worker requires an output directory.")
        try:
            run_worker(args._worker, args._output_dir, args.device, args.vggt_root, args.checkpoint)
            return 0
        except Exception as error:
            traceback.print_exc()
            write_json(args._output_dir / "error.json", dict(status="failed", error_type=type(error).__name__, message=str(error)))
            return 1
    try:
        files = discover_inputs(args.inputs, args.input_dir)
        if args.validate_only:
            from vggt_inputs import load_clip
            for path in files:
                clip = load_clip(path)
                print(f"Valid: {path.name} | {clip['frames_rgb'].shape} | {len(clip['point_ids'])} annotated points")
                del clip
            return 0
        return run_batch(files, args)
    except KeyboardInterrupt:
        print("Interrupted; active worker terminated.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
