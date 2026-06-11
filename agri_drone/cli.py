"""Command-line entry point.

Examples:
    python -m agri_drone.cli extract  --video field.mp4 --models models --output dataset
    python -m agri_drone.cli generate --kind gan --out synthetic --per-class 32
    python -m agri_drone.cli generate --kind vae --out synthetic_healthy --num 64
    python -m agri_drone.cli growth   --metadata dataset/metadata.csv --out outputs
"""
from __future__ import annotations

import argparse
import logging

from .config import Config


def _cmd_extract(args) -> None:
    from .process_video import process_video

    cfg = Config()
    cfg.detector_path = f"{args.models}/yolo_detector.pt"
    cfg.classifier_path = f"{args.models}/yolo_classifier.pt"
    cfg.leaf_seg_path = f"{args.models}/leaf_seg.pth"
    cfg.disease_seg_path = f"{args.models}/diseases_leaf_segmentation.pth"
    cfg.frame_skip = args.frame_skip
    cfg.gsd_cm_per_px = args.gsd
    process_video(args.video, cfg, output_dir=args.output,
                  video_out=None if args.no_video else args.video_out,
                  display=args.display)


def _cmd_generate(args) -> None:
    import torch
    from . import models
    from .generate import generate_gan_images, generate_vae_images

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.kind == "gan":
        cfg.gan_path = args.weights or cfg.gan_path
        gen = models.build_generator(cfg, device)
        n = generate_gan_images(gen, cfg.gan_num_classes, args.per_class, args.out,
                                cfg.gan_latent_dim, device)
    else:
        cfg.vae_path = args.weights or cfg.vae_path
        vae = models.build_vae(cfg, device)
        n = generate_vae_images(vae, args.num, args.out, cfg.vae_latent_dim, device)
    print(f"Saved {n} individual images to {args.out}")


def _cmd_growth(args) -> None:
    from .growth import run
    run(args.metadata, out_dir=args.out, labels_csv=args.labels)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agri_drone")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="process a drone video into per-leaf data")
    e.add_argument("--video", required=True)
    e.add_argument("--models", default="models")
    e.add_argument("--output", default="dataset")
    e.add_argument("--video-out", default="output_annotated.mp4")
    e.add_argument("--no-video", action="store_true", help="do not write the annotated video")
    e.add_argument("--frame-skip", type=int, default=3)
    e.add_argument("--gsd", type=float, default=0.05, help="ground sampling distance cm/pixel")
    e.add_argument("--display", action="store_true")
    e.set_defaults(func=_cmd_extract)

    g = sub.add_parser("generate", help="generate synthetic images (one file per image)")
    g.add_argument("--kind", choices=["gan", "vae"], required=True)
    g.add_argument("--weights")
    g.add_argument("--out", default="synthetic")
    g.add_argument("--per-class", type=int, default=32, help="GAN: images per class")
    g.add_argument("--num", type=int, default=64, help="VAE: total images")
    g.set_defaults(func=_cmd_generate)

    w = sub.add_parser("growth", help="crop-growth feature engineering + regression")
    w.add_argument("--metadata", required=True)
    w.add_argument("--out", default="outputs")
    w.add_argument("--labels", help="CSV with columns: frame,<target> (real ground truth)")
    w.set_defaults(func=_cmd_growth)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
