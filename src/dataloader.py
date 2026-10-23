"""Dataset loader and inspector for Fruit Freshness and Quality Prediction CBP.

Handles dataset discovery, folder validation, corruption filtering,
and reproducible train/validation/test splitting without data leakage.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import config


class FruitDataLoader:
    """Discovers, validates, and partitions real fruit freshness image datasets."""

    def __init__(self, data_dir: Optional[Path] = None, seed: int = config.RANDOM_SEED):
        self.data_dir = Path(data_dir) if data_dir else config.RAW_DATA_DIR
        self.seed = seed
        self.classes: List[str] = []
        self.samples_df: Optional[pd.DataFrame] = None

    def discover_and_validate(self) -> pd.DataFrame:
        """Scan the dataset directory, validate images, and assemble metadata."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory '{self.data_dir}' does not exist. "
                f"Please place your dataset in '{config.RAW_DATA_DIR}'."
            )

        records = []
        corrupted_count = 0
        empty_classes = []

        # Check subdirectories
        subdirs = [d for d in sorted(self.data_dir.iterdir()) if d.is_dir() and not d.name.startswith(".")]

        # Also support pre-split train/test layouts if present
        if any(d.name.lower() in ("train", "test", "validation") for d in subdirs):
            print(f"[DataLoader] Detected partitioned layout in {self.data_dir}. Scanning subfolders...")
            for split_dir in subdirs:
                if split_dir.is_dir():
                    for class_dir in sorted(split_dir.iterdir()):
                        if class_dir.is_dir() and not class_dir.name.startswith("."):
                            cls_name = class_dir.name
                            img_files = [
                                p for p in class_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
                            ]
                            for img_path in img_files:
                                if self._is_valid_image(img_path):
                                    records.append({
                                        "image_path": str(img_path.resolve()),
                                        "filename": img_path.name,
                                        "class": cls_name,
                                        "partition": split_dir.name.lower()
                                    })
                                else:
                                    corrupted_count += 1
        else:
            # Flat class-directory layout
            for class_dir in subdirs:
                cls_name = class_dir.name
                img_files = [
                    p for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
                ]
                if not img_files:
                    empty_classes.append(cls_name)
                    continue

                for img_path in img_files:
                    if self._is_valid_image(img_path):
                        records.append({
                            "image_path": str(img_path.resolve()),
                            "filename": img_path.name,
                            "class": cls_name,
                            "partition": None
                        })
                    else:
                        corrupted_count += 1

        if not records:
            raise ValueError(
                f"No valid image files found in '{self.data_dir}'. "
                f"Supported formats: {config.SUPPORTED_EXTENSIONS}"
            )

        df = pd.DataFrame(records)
        self.classes = sorted(df["class"].unique().tolist())
        self.samples_df = df

        print("==================================================")
        print(f"Dataset Loaded Successfully: {self.data_dir}")
        print(f"Total Valid Images Discovered: {len(df)}")
        print(f"Total Classes Identified: {len(self.classes)}")
        if corrupted_count > 0:
            print(f"Corrupted/Unreadable Images Filtered: {corrupted_count}")
        if empty_classes:
            print(f"Warning - Empty folders skipped: {empty_classes}")
        print("--------------------------------------------------")
        print("Class Distribution:")
        counts = df["class"].value_counts()
        for cls, count in counts.items():
            print(f"  - {cls:<22} : {count:>5} images")
        print("==================================================")

        return df

    @staticmethod
    def _is_valid_image(image_path: Path) -> bool:
        """Check if image file is non-empty and decodable."""
        if not image_path.exists() or image_path.stat().st_size == 0:
            return False
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def get_splits(
        self,
        train_ratio: float = config.TRAIN_RATIO,
        val_ratio: float = config.VAL_RATIO,
        test_ratio: float = config.TEST_RATIO,
        max_samples_per_class: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified, reproducible train/validation/test splits.

        Prevents duplicate leakage by asserting disjoint index sets.
        Optionally caps max_samples_per_class for rapid CPU training.
        """
        if self.samples_df is None:
            self.discover_and_validate()

        df = self.samples_df.copy()

        # If already partitioned by train/val/test folders
        if df["partition"].notna().any() and "train" in df["partition"].values:
            train_df = df[df["partition"] == "train"].reset_index(drop=True)
            test_df = df[df["partition"].isin(["test", "val", "validation"])].reset_index(drop=True)
            if len(test_df) == 0:
                train_df, test_df = train_test_split(
                    train_df, test_size=0.2, stratify=train_df["class"], random_state=self.seed
                )
            val_df, test_df = train_test_split(
                test_df, test_size=0.5, stratify=test_df["class"], random_state=self.seed
            )
            return train_df, val_df, test_df

        # If user wants a balanced subset for fast demonstration / testing
        if max_samples_per_class is not None and max_samples_per_class > 0:
            sampled_dfs = []
            for cls in self.classes:
                cls_subset = df[df["class"] == cls]
                n_take = min(len(cls_subset), max_samples_per_class)
                sampled_dfs.append(cls_subset.sample(n=n_take, random_state=self.seed))
            df = pd.concat(sampled_dfs, ignore_index=True)

        # Standard stratified two-step split
        test_size = test_ratio / (train_ratio + val_ratio + test_ratio)
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["class"],
            random_state=self.seed
        )

        val_rel_size = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_rel_size,
            stratify=train_val_df["class"],
            random_state=self.seed
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Assert no data leakage across splits
        train_paths = set(train_df["image_path"])
        val_paths = set(val_df["image_path"])
        test_paths = set(test_df["image_path"])
        assert len(train_paths.intersection(val_paths)) == 0, "Data leakage detected between train and val!"
        assert len(train_paths.intersection(test_paths)) == 0, "Data leakage detected between train and test!"
        assert len(val_paths.intersection(test_paths)) == 0, "Data leakage detected between val and test!"

        print(f"[Splits] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        return train_df, val_df, test_df


def inspect_dataset(data_dir: Optional[str] = None):
    """CLI utility function for manual dataset inspection."""
    path = Path(data_dir) if data_dir else config.RAW_DATA_DIR
    loader = FruitDataLoader(path)
    df = loader.discover_and_validate()
    loader.get_splits()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and split fruit dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.RAW_DATA_DIR),
        help="Path to fruit dataset directory."
    )
    args = parser.parse_args()
    inspect_dataset(args.data_dir)
