from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


SDXL_MODEL_ID = "stabilityai/sdxl-turbo"
SAE_REPO_ID = "Goodfire/SDXL-Turbo-SAE-ldown.attns.2.1"
SAE_FILENAME = "SDXL-Turbo-SAE-ldown_blocks.2.attentions.1.pt"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_FEATURE_IDS = [770, 4071, 4443, 6214, 7446]
DEFAULT_SEEDS = [11, 37, 89]
DEFAULT_CPU_WORKERS = 8


@dataclass
class FeatureRun:
    feature_id: int
    seed: int
    image_path: str


@dataclass
class StageSummary:
    stage: str
    item_count: int
    batch_size: int
    elapsed_seconds: float
    items_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate the SAE feature autolabelling pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--output-dir",
            type=Path,
            default=Path("autolabel_pipeline/runs/latest"),
            help="Directory for images, manifests, labels, and profiling logs.",
        )
        p.add_argument(
            "--feature-ids",
            type=int,
            nargs="+",
            default=DEFAULT_FEATURE_IDS,
            help="Feature IDs to process.",
        )

    generate = subparsers.add_parser("generate", help="Generate 3 steered images per SAE feature.")
    add_shared_flags(generate)
    generate.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds per feature.")
    generate.add_argument(
        "--steering-scale",
        type=float,
        default=120.0,
        help="Coefficient applied to the SAE decoder direction at down_blocks[2].attentions[1].",
    )
    generate.add_argument("--height", type=int, default=512)
    generate.add_argument("--width", type=int, default=512)
    generate.add_argument(
        "--generate-batch-size",
        type=int,
        default=24,
        help="Number of feature/seed jobs to generate in parallel.",
    )
    generate.add_argument(
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="Number of CPU workers for image save/load work.",
    )

    label = subparsers.add_parser("label", help="Label generated image triplets with Qwen2.5-VL-7B-Instruct.")
    add_shared_flags(label)
    label.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit path to the generation manifest JSON.",
    )
    label.add_argument("--max-new-tokens", type=int, default=96)
    label.add_argument(
        "--label-batch-size",
        type=int,
        default=8,
        help="Number of feature triplets to label in parallel.",
    )
    label.add_argument(
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="Number of CPU workers for image save/load work.",
    )

    run = subparsers.add_parser("run", help="Run staged generation then labelling in separate subprocesses.")
    add_shared_flags(run)
    run.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    run.add_argument("--steering-scale", type=float, default=120.0)
    run.add_argument("--height", type=int, default=512)
    run.add_argument("--width", type=int, default=512)
    run.add_argument("--max-new-tokens", type=int, default=96)
    run.add_argument("--generate-batch-size", type=int, default=24)
    run.add_argument("--label-batch-size", type=int, default=8)
    run.add_argument("--cpu-workers", type=int, default=DEFAULT_CPU_WORKERS)

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def snapshot_gpu(output_dir: Path, tag: str) -> None:
    ensure_dir(output_dir)
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    line = result.stdout.strip().splitlines()[0]
    timestamp, name, total_mem, used_mem, util = [part.strip() for part in line.split(",")]
    record = {
        "tag": tag,
        "timestamp": timestamp,
        "name": name,
        "memory_total_mib": int(total_mem),
        "memory_used_mib": int(used_mem),
        "utilization_gpu_pct": int(util),
    }
    with (output_dir / "gpu_profile.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def write_stage_summary(output_dir: Path, summary: StageSummary) -> None:
    with (output_dir / "stage_summary.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(summary)) + "\n")


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def save_image(image: Image.Image, image_path: str) -> None:
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    image.close()


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_decoder_matrix() -> torch.Tensor:
    sae_path = hf_hub_download(repo_id=SAE_REPO_ID, filename=SAE_FILENAME, repo_type="model")
    state_dict = torch.load(sae_path, map_location="cpu", weights_only=True)
    return state_dict["decoder.weight"].contiguous()


class SteeringHook:
    def __init__(self, module: torch.nn.Module, vector: torch.Tensor):
        self.module = module
        self.vector = vector
        self.handle: Any | None = None

    def __enter__(self) -> "SteeringHook":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.vector.ndim == 1:
            steering = self.vector.view(1, -1, 1, 1)
        elif self.vector.ndim == 2:
            steering = self.vector.view(self.vector.shape[0], self.vector.shape[1], 1, 1)
        elif self.vector.ndim == 3:
            steering = self.vector.unsqueeze(0)
        elif self.vector.ndim == 4:
            steering = self.vector
        else:
            raise ValueError(f"Unsupported steering tensor rank: {self.vector.ndim}")
        if hasattr(output, "sample"):
            output.sample = output.sample + steering
            return output
        if isinstance(output, tuple):
            return (output[0] + steering, *output[1:])
        return output + steering


def build_sdxl_pipeline() -> AutoPipelineForText2Image:
    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        variant="fp16",
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()
    pipe.enable_attention_slicing()
    return pipe


def generate_images(args: argparse.Namespace) -> Path:
    ensure_dir(args.output_dir)
    images_dir = args.output_dir / "images"
    ensure_dir(images_dir)

    snapshot_gpu(args.output_dir, "generate:start")
    decoder_weight = load_decoder_matrix()
    pipe = build_sdxl_pipeline()
    snapshot_gpu(args.output_dir, "generate:models_loaded")

    runs: list[FeatureRun] = []
    target_module = pipe.unet.down_blocks[2].attentions[1]
    jobs: list[tuple[int, int, Path]] = []
    for feature_id in args.feature_ids:
        feature_dir = images_dir / f"feature_{feature_id}"
        ensure_dir(feature_dir)
        for seed in args.seeds:
            jobs.append((feature_id, seed, feature_dir / f"seed_{seed}.png"))

    start_time = time.perf_counter()
    save_futures: list[Future[Any]] = []
    with ThreadPoolExecutor(max_workers=args.cpu_workers) as executor:
        for batch_index, batch_jobs in enumerate(chunked(jobs, args.generate_batch_size), start=1):
            feature_ids = [feature_id for feature_id, _seed, _path in batch_jobs]
            basis_vectors = (
                decoder_weight[:, feature_ids].T.to(device="cuda", dtype=pipe.unet.dtype) * args.steering_scale
            )
            basis_vectors = decoder_weight[:, feature_ids].T.to(device="cuda", dtype=pipe.unet.dtype)
            basis_vectors = basis_vectors / basis_vectors.norm(dim=1, keepdim=True).clamp_min(1e-8)
            basis_vectors = basis_vectors * args.steering_scale

            generators = [torch.Generator(device="cuda").manual_seed(seed) for _feature_id, seed, _path in batch_jobs]
            prompts = [""] * len(batch_jobs)

            with SteeringHook(target_module, basis_vectors):
                result = pipe(
                    prompt=prompts,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    generator=generators,
                    height=args.height,
                    width=args.width,
                )

            for image, (feature_id, seed, image_path) in zip(result.images, batch_jobs):
                save_futures.append(executor.submit(save_image, image, str(image_path)))
                runs.append(FeatureRun(feature_id=feature_id, seed=seed, image_path=str(image_path)))

            if len(save_futures) >= args.cpu_workers * 4:
                for future in save_futures[: args.cpu_workers * 2]:
                    future.result()
                save_futures = save_futures[args.cpu_workers * 2 :]

            snapshot_gpu(args.output_dir, f"generate:batch_{batch_index}")

        for future in save_futures:
            future.result()

    elapsed = time.perf_counter() - start_time
    write_stage_summary(
        args.output_dir,
        StageSummary(
            stage="generate",
            item_count=len(jobs),
            batch_size=args.generate_batch_size,
            elapsed_seconds=elapsed,
            items_per_second=len(jobs) / elapsed if elapsed else 0.0,
        ),
    )

    manifest = {
        "model_id": SDXL_MODEL_ID,
        "sae_repo_id": SAE_REPO_ID,
        "sae_filename": SAE_FILENAME,
        "steering_scale": args.steering_scale,
        "height": args.height,
        "width": args.width,
        "generate_batch_size": args.generate_batch_size,
        "feature_runs": [asdict(run) for run in runs],
    }
    manifest_path = args.output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    del pipe
    del decoder_weight
    cleanup_cuda()
    snapshot_gpu(args.output_dir, "generate:after_unload")
    return manifest_path


def load_triplets(manifest_path: Path, feature_ids: list[int]) -> dict[int, list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[int, list[Path]] = {}
    for item in manifest["feature_runs"]:
        feature_id = int(item["feature_id"])
        if feature_id not in feature_ids:
            continue
        grouped.setdefault(feature_id, []).append(Path(item["image_path"]))
    for feature_id, paths in grouped.items():
        grouped[feature_id] = sorted(paths)
    return grouped


def build_qwen_model() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, processor


def build_label_prompt() -> str:
    return (
        "You are labelling one SAE feature from a diffusion model. "
        "You will see three unconditional feature-steered generations from different random seeds. "
        "Infer the recurring visual concept shared across the images. "
        "Return exactly one JSON object with keys "
        '"label" and "reason". '
        'Use a short label of 1 to 6 words. '
        'If the concept is inconsistent, use "unclear visual concept".'
    )


def extract_json_blob(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"label": text.strip().splitlines()[0].strip(), "reason": text.strip()}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"label": text.strip().splitlines()[0].strip(), "reason": text.strip()}


def label_images(args: argparse.Namespace) -> Path:
    ensure_dir(args.output_dir)
    manifest_path = args.manifest_path or (args.output_dir / "generation_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Generation manifest not found at {manifest_path}")

    feature_triplets = load_triplets(manifest_path, args.feature_ids)
    snapshot_gpu(args.output_dir, "label:start")
    model, processor = build_qwen_model()
    snapshot_gpu(args.output_dir, "label:model_loaded")

    labels: list[dict[str, Any]] = []
    prompt = build_label_prompt()
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.cpu_workers) as executor:
        for batch_index, feature_batch in enumerate(chunked(args.feature_ids, args.label_batch_size), start=1):
            text_batch: list[str] = []
            load_jobs = [feature_triplets[feature_id] for feature_id in feature_batch]
            image_batch = list(executor.map(lambda paths: [load_rgb_image(path) for path in paths], load_jobs))

            for _images in image_batch:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text_batch.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            flat_images = [image for images in image_batch for image in images]
            inputs = processor(text=text_batch, images=flat_images, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for feature_id, image_paths, response in zip(feature_batch, [feature_triplets[f] for f in feature_batch], responses):
                parsed = extract_json_blob(response)
                labels.append(
                    {
                        "feature_id": feature_id,
                        "label": parsed.get("label", "").strip(),
                        "reason": parsed.get("reason", "").strip(),
                        "raw_response": response.strip(),
                        "image_paths": [str(path) for path in image_paths],
                    }
                )
            for images in image_batch:
                for image in images:
                    image.close()
            snapshot_gpu(args.output_dir, f"label:batch_{batch_index}")

    elapsed = time.perf_counter() - start_time
    write_stage_summary(
        args.output_dir,
        StageSummary(
            stage="label",
            item_count=len(args.feature_ids),
            batch_size=args.label_batch_size,
            elapsed_seconds=elapsed,
            items_per_second=len(args.feature_ids) / elapsed if elapsed else 0.0,
        ),
    )

    labels_path = args.output_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_id", "label", "reason"])
        writer.writeheader()
        for row in labels:
            writer.writerow({key: row[key] for key in ["feature_id", "label", "reason"]})

    del model
    del processor
    cleanup_cuda()
    snapshot_gpu(args.output_dir, "label:after_unload")
    return labels_path


def run_staged(args: argparse.Namespace) -> None:
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
    ]
    feature_args = ["--feature-ids", *[str(x) for x in args.feature_ids]]
    output_args = ["--output-dir", str(args.output_dir)]

    generate_cmd = base + [
        "generate",
        *output_args,
        *feature_args,
        "--seeds",
        *[str(x) for x in args.seeds],
        "--steering-scale",
        str(args.steering_scale),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--generate-batch-size",
        str(args.generate_batch_size),
        "--cpu-workers",
        str(args.cpu_workers),
    ]
    subprocess.run(generate_cmd, check=True)

    label_cmd = base + [
        "label",
        *output_args,
        *feature_args,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--label-batch-size",
        str(args.label_batch_size),
        "--cpu-workers",
        str(args.cpu_workers),
    ]
    subprocess.run(label_cmd, check=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(args.cpu_workers, DEFAULT_CPU_WORKERS))
    if args.command == "generate":
        manifest_path = generate_images(args)
        print(f"Wrote generation manifest to {manifest_path}")
    elif args.command == "label":
        labels_path = label_images(args)
        print(f"Wrote labels to {labels_path}")
    elif args.command == "run":
        run_staged(args)
        print(f"Completed staged run in {args.output_dir}")
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
