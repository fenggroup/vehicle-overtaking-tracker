"""
Export RT-DETR (PaddleDetection) model to ONNX format.

This script:
1. Clones PaddleDetection repository (if not cached)
2. Exports static inference model using provided config and weights
3. Converts exported Paddle model to ONNX using paddle2onnx

Example:
    python scripts/export_rtdetr_onnx.py \
        --weights models/rtdetr_l/model.pdparams \
        --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        --image-size 640 640 \
        --device cpu \
        --out rtdetr.onnx

Requirements:
    pip install paddlepaddle paddledet paddle2onnx onnx
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    """Execute shell command and stream output."""
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    for line in iter(process.stdout.readline, b""):
        sys.stdout.write(line.decode("utf-8", errors="ignore"))
    process.stdout.close()
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {cmd}")


def ensure_paddledetection(cache_dir: Path) -> Path:
    """Clone PaddleDetection repository if not already present."""
    repo_dir = cache_dir / "PaddleDetection"
    if repo_dir.exists():
        return repo_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    run("git clone https://github.com/PaddlePaddle/PaddleDetection.git", cwd=str(cache_dir))
    return repo_dir


def export_paddle_infer(repo_dir: Path, config_rel: str, weights_path: Path, img_h: int, img_w: int, use_gpu: bool):
    """Export PaddleDetection model to inference format."""
    image_shape_arg = f"TestReader.inputs_def.image_shape=[3,{img_h},{img_w}]"
    gpu_flag = "use_gpu=True" if use_gpu else "use_gpu=False"
    cmd = (
        f"python tools/export_model.py -c {config_rel} "
        f"-o weights={weights_path.as_posix()} {image_shape_arg} {gpu_flag}"
    )
    run(cmd, cwd=str(repo_dir))


def find_infer_dir(repo_dir: Path) -> Path:
    """Find the most recently created inference export directory."""
    out = repo_dir / "output_inference"
    if not out.exists():
        raise FileNotFoundError("output_inference directory not found after export")
    subdirs = [p for p in out.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError("No exported model directory under output_inference")
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    pdmodel = latest / "inference.pdmodel"
    pdparams = latest / "inference.pdiparams"
    if not pdmodel.exists() or not pdparams.exists():
        raise FileNotFoundError("inference.pdmodel or inference.pdiparams missing in export directory")
    return latest


def convert_to_onnx(model_dir: Path, save_file: Path, opset: int):
    """Convert Paddle inference model to ONNX format."""
    cmd = (
        f"paddle2onnx --model_dir {model_dir.as_posix()} "
        f"--model_filename inference.pdmodel "
        f"--params_filename inference.pdiparams "
        f"--save_file {save_file.as_posix()} --opset_version {opset} --enable_onnx_checker True"
    )
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Export RT-DETR (Paddle) to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pdparams weights file")
    parser.add_argument("--config", default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml", help="Config path relative to PaddleDetection root")
    parser.add_argument("--image-size", nargs=2, type=int, default=[640, 640], help="Export image size (height width)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Export device")
    parser.add_argument("--out", required=True, help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--cache-dir", default=".cache_pd", help="Directory to cache PaddleDetection repository")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    cache_dir = Path(args.cache_dir).resolve()
    repo_dir = ensure_paddledetection(cache_dir)

    img_h, img_w = args.image_size
    use_gpu = args.device == "gpu"

    export_paddle_infer(repo_dir, args.config, weights_path, img_h, img_w, use_gpu)
    infer_dir = find_infer_dir(repo_dir)

    onnx_out = Path(args.out).resolve()
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    convert_to_onnx(infer_dir, onnx_out, args.opset)

    print(f"Export completed: {onnx_out}")


if __name__ == "__main__":
    main()
