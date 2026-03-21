#!/usr/bin/env python3
"""
Training entry point for the Brain Tumor AI model.

Usage:
  # From inside a container:
  python -m app.train

  # Or with docker compose:
  docker compose exec backend python -m app.train

  # Or run the full pipeline (download + train):
  docker compose exec backend python -m app.train --download

  # Or via the API:
  curl -X POST http://localhost:8000/api/v1/training/start
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train the Brain Tumor Classifier")
    parser.add_argument(
        "--download", action="store_true",
        help="Download datasets before training (requires Kaggle API credentials)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="/data",
        help="Root data directory (default: /data)",
    )
    parser.add_argument(
        "--model-path", type=str, default="/models/brain_tumor_2d.pth",
        help="Path to save trained model (default: /models/brain_tumor_2d.pth)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--store-graph", action="store_true",
        help="Store dataset metadata and training results in FalkorDB",
    )
    args = parser.parse_args()

    # Step 1: Dataset preparation
    if args.download:
        logger.info("=== Step 1: Downloading datasets ===")
        from app.dataset.downloader import DatasetDownloader

        downloader = DatasetDownloader(args.data_dir)
        results = downloader.prepare_dataset()
        logger.info(f"Dataset preparation complete: {results}")
    else:
        logger.info("Skipping download (use --download to download datasets)")

    # Step 2: Check dataset exists
    from pathlib import Path
    combined_dir = Path(args.data_dir) / "combined"
    train_dir = combined_dir / "train"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        logger.error(
            f"No training data found at {train_dir}.\n"
            "Please either:\n"
            "  1. Run with --download flag (requires Kaggle credentials)\n"
            "  2. Manually place dataset in /data/combined/train/{glioma,meningioma,pituitary,no_tumor}/\n"
            "  3. Download from: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri\n"
            "     Extract and place in /data/combined/ with train/ and val/ subdirectories"
        )
        sys.exit(1)

    # Step 3: Store dataset metadata in FalkorDB
    if args.store_graph:
        logger.info("=== Storing dataset metadata in FalkorDB ===")
        try:
            from app.services.graph_db import get_falkordb

            graph_db = get_falkordb()
            graph_db.initialize_schema()

            images = []
            for split in ["train", "val"]:
                split_dir = combined_dir / split
                if not split_dir.exists():
                    continue
                for cls_dir in split_dir.iterdir():
                    if cls_dir.is_dir():
                        for img_path in cls_dir.iterdir():
                            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                                images.append({
                                    "path": str(img_path),
                                    "class_label": cls_dir.name,
                                    "split": split,
                                })

            graph_db.store_dataset_metadata(images, "combined")
            logger.info(f"Stored {len(images)} images in FalkorDB")
        except Exception as e:
            logger.warning(f"FalkorDB storage failed (non-fatal): {e}")

    # Step 4: Train all 4 models (ensemble)
    logger.info("=== Training Ensemble (4 models) ===")
    from app.dataset.trainer import train_all_models

    all_results = train_all_models(
        data_dir=str(combined_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
    )

    logger.info("=== Ensemble Training Complete ===")
    for model_name, res in all_results.items():
        logger.info(f"  {model_name}: accuracy={res['best_val_accuracy']:.4f}, epochs={res['epochs_trained']}, saved={res['model_path']}")

    best_acc = max(r["best_val_accuracy"] for r in all_results.values())
    logger.info(f"Best individual model accuracy: {best_acc:.4f}")

    # Store each training result in FalkorDB
    if args.store_graph:
        try:
            import uuid
            from app.services.graph_db import get_falkordb
            from app.dataset.downloader import DatasetDownloader

            graph_db = get_falkordb()
            downloader = DatasetDownloader(args.data_dir)
            stats = downloader.get_dataset_stats()

            for model_name, res in all_results.items():
                graph_db.store_training_run(
                    run_id=str(uuid.uuid4()),
                    accuracy=res["best_val_accuracy"],
                    epochs=res["epochs_trained"],
                    model_path=res["model_path"],
                    class_distribution={**stats, "model_name": model_name},
                )
            logger.info("All training results stored in FalkorDB")
        except Exception as e:
            logger.warning(f"FalkorDB storage failed (non-fatal): {e}")

    if best_acc >= 0.90:
        logger.info("SUCCESS: Ensemble models achieved >90% validation accuracy!")
        logger.info("With ensemble + TTA, combined inference targets >98% confidence.")
    else:
        logger.warning(
            f"Best model accuracy is {best_acc:.1%}. "
            "Consider: more data, longer training, or hyperparameter tuning."
        )


if __name__ == "__main__":
    main()
