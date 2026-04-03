from __future__ import annotations

import argparse
import json
from pathlib import Path

import clusterfun as clt
import pandas as pd
import torch
import umap
from huggingface_hub import hf_hub_download


SAE_REPO_ID = "Goodfire/SDXL-Turbo-SAE-ldown.attns.2.1"
SAE_FILENAME = "SDXL-Turbo-SAE-ldown_blocks.2.attentions.1.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a UMAP scatter plot for labelled SAE features using clusterfun.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("autolabel_pipeline/runs/benchmark_1024_bs160_32"),
        help="Directory containing labels.json from the autolabelling benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("autolabel_pipeline/plots/benchmark_1024_umap"),
        help="Directory for the UMAP dataframe and clusterfun metadata.",
    )
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--min-dist", type=float, default=0.08)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_decoder_vectors(feature_ids: list[int]) -> torch.Tensor:
    sae_path = hf_hub_download(repo_id=SAE_REPO_ID, filename=SAE_FILENAME, repo_type="model")
    state_dict = torch.load(sae_path, map_location="cpu", weights_only=True)
    decoder = state_dict["decoder.weight"].T
    return decoder[feature_ids].float()


def build_dataframe(run_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    labels = json.loads((run_dir / "labels.json").read_text(encoding="utf-8"))
    feature_ids = [int(row["feature_id"]) for row in labels]
    decoder_vectors = load_decoder_vectors(feature_ids)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(decoder_vectors.numpy())

    rows = []
    for idx, row in enumerate(labels):
        image_path = Path(row["image_paths"][0]).resolve()
        rows.append(
            {
                "feature_id": int(row["feature_id"]),
                "label": row["label"],
                "reason": row.get("reason", ""),
                "preview_image": str(image_path),
                "umap_x": float(embedding[idx, 0]),
                "umap_y": float(embedding[idx, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_id").reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataframe(args.run_dir, args)
    dataframe_path = args.output_dir / "umap_points.csv"
    df.to_csv(dataframe_path, index=False)

    plot_path = clt.scatter(
        df,
        x="umap_x",
        y="umap_y",
        media="preview_image",
        title=f"UMAP of {len(df)} labelled SAE features",
        display=["label", "feature_id", "reason"],
        show=False,
    )

    metadata = {
        "run_dir": str(args.run_dir.resolve()),
        "dataframe_csv": str(dataframe_path.resolve()),
        "clusterfun_cache_dir": str(plot_path.resolve()),
        "clusterfun_uuid": plot_path.name,
        "serve_command": f"./.venv/bin/clusterfun {plot_path.name}",
    }
    (args.output_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
