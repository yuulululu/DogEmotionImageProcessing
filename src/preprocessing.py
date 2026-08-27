"""
preprocessing.py - Data Preprocessing Pipeline สำหรับ Dog Emotion Dataset
==========================================================================
ขั้นตอนการ preprocess:
  1. ดาวน์โหลด/ค้นหา dataset จาก kagglehub อัตโนมัติ
  2. ลบ corrupted images
  3. ตรวจจับและจัดการภาพซ้ำ (duplicate) ด้วย pHash
  4. จัดการ class imbalance ด้วย oversampling
  5. แปลง format → .jpg  และ color space → RGB ให้เป็นมาตรฐานเดียวกัน

ผลลัพธ์: โฟลเดอร์ dataset ที่ผ่านการ preprocess พร้อมใช้งาน
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import Counter

import kagglehub
import numpy as np
from PIL import Image, UnidentifiedImageError

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET_SLUG = "mohitagarwal17/dog-emotion-datasetcleaned-version"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════
def download_dataset() -> Path:
    """ดาวน์โหลด dataset จาก kagglehub และคืน path"""
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"[INFO] Dataset path: {path}")
    return Path(path)


def scan_images(root: Path) -> dict[str, list[Path]]:
    """
    สแกนหารูปภาพทั้งหมดใต้ root อัตโนมัติ
    คืน dict: { label_name: [path1, path2, ...] }
    """
    class_images: dict[str, list[Path]] = {}
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() in IMAGE_EXTENSIONS and img_path.is_file():
            label = img_path.parent.name
            class_images.setdefault(label, []).append(img_path)
    return class_images


def phash(img: Image.Image, hash_size: int = 8) -> str:
    """
    คำนวณ Perceptual Hash (pHash) ของภาพ
    ──────────────────────────────────────
    1. Resize เป็น (hash_size*4, hash_size*4) แล้วแปลงเป็น grayscale
    2. คำนวณ DCT (Discrete Cosine Transform)
    3. ตัดเอาเฉพาะ top-left (low-frequency) ของ DCT
    4. เปรียบเทียบกับ median เพื่อสร้าง binary hash
    """
    # Resize + grayscale
    img_resized = img.convert("L").resize(
        (hash_size * 4, hash_size * 4), Image.LANCZOS
    )
    pixels = np.array(img_resized, dtype=np.float64)

    # Simple DCT-like transform (เพื่อไม่ต้องพึ่ง scipy)
    # ใช้ 2D DCT ผ่าน matrix multiplication
    dct_matrix = _dct_matrix(hash_size * 4)
    dct_result = dct_matrix @ pixels @ dct_matrix.T

    # ตัดเฉพาะ top-left block (low frequency)
    low_freq = dct_result[:hash_size, :hash_size]

    # สร้าง hash จาก median
    median_val = np.median(low_freq)
    diff = low_freq > median_val

    # แปลงเป็น hex string
    hash_bits = diff.flatten()
    hash_int = 0
    for bit in hash_bits:
        hash_int = (hash_int << 1) | int(bit)
    return format(hash_int, f"0{hash_size * hash_size // 4}x")


def _dct_matrix(n: int) -> np.ndarray:
    """สร้าง DCT-II matrix ขนาด n×n"""
    mat = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            if k == 0:
                mat[k, i] = 1.0 / np.sqrt(n)
            else:
                mat[k, i] = np.sqrt(2.0 / n) * np.cos(
                    np.pi * (2 * i + 1) * k / (2 * n)
                )
    return mat


def hamming_distance(hash1: str, hash2: str) -> int:
    """คำนวณ Hamming Distance ระหว่าง 2 hash strings"""
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count("1")


# ══════════════════════════════════════════════
# STEP 1: ลบ Corrupted Images
# ══════════════════════════════════════════════
def remove_corrupted(class_images: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """ตรวจสอบและลบไฟล์ภาพที่เสียหายออกจากรายการ"""
    print(f"\n{'='*60}")
    print("  STEP 1: Remove Corrupted Images")
    print(f"{'='*60}")

    cleaned: dict[str, list[Path]] = {}
    total_removed = 0

    for label, paths in class_images.items():
        valid = []
        removed = 0
        for p in paths:
            try:
                img = Image.open(p)
                img.verify()
                # verify() ทำให้ img ใช้ต่อไม่ได้ ต้องเปิดใหม่เพื่อทดสอบ load
                img = Image.open(p)
                img.load()
                valid.append(p)
            except (UnidentifiedImageError, OSError, SyntaxError, Exception) as e:
                print(f"    ✗ CORRUPTED: {p.name}  ({e.__class__.__name__})")
                removed += 1
        cleaned[label] = valid
        total_removed += removed

    total_remaining = sum(len(v) for v in cleaned.values())
    print(f"\n  → Removed: {total_removed} corrupted")
    print(f"  → Remaining: {total_remaining:,} images")
    return cleaned


# ══════════════════════════════════════════════
# STEP 2: ตรวจจับและจัดการภาพซ้ำด้วย pHash
# ══════════════════════════════════════════════
def remove_duplicates(
    class_images: dict[str, list[Path]],
    threshold: int = 6,
) -> dict[str, list[Path]]:
    """
    ตรวจจับภาพซ้ำด้วย pHash
    - ภาพที่มี hamming distance <= threshold ถือว่าซ้ำ
    - เก็บเฉพาะภาพแรกในแต่ละกลุ่มที่ซ้ำ
    """
    print(f"\n{'='*60}")
    print(f"  STEP 2: Remove Duplicates (pHash, threshold={threshold})")
    print(f"{'='*60}")

    deduped: dict[str, list[Path]] = {}
    total_removed = 0

    for label, paths in class_images.items():
        # คำนวณ pHash ทุกภาพ
        hashes: list[tuple[str, Path]] = []
        for p in paths:
            try:
                img = Image.open(p)
                h = phash(img)
                hashes.append((h, p))
            except Exception:
                hashes.append(("", p))

        # หาภาพซ้ำ
        keep = []
        removed_in_class = 0
        seen_hashes: list[str] = []

        for h, p in hashes:
            if h == "":
                keep.append(p)  # hash ไม่ได้ → เก็บไว้ก่อน
                continue

            is_dup = False
            for seen in seen_hashes:
                if hamming_distance(h, seen) <= threshold:
                    is_dup = True
                    break

            if is_dup:
                removed_in_class += 1
            else:
                seen_hashes.append(h)
                keep.append(p)

        deduped[label] = keep
        total_removed += removed_in_class
        if removed_in_class > 0:
            print(f"    [{label}] removed {removed_in_class} duplicates "
                  f"({len(paths)} → {len(keep)})")

    total_remaining = sum(len(v) for v in deduped.values())
    print(f"\n  → Removed: {total_removed} duplicates total")
    print(f"  → Remaining: {total_remaining:,} images")
    return deduped


# ══════════════════════════════════════════════
# STEP 3: จัดการ Class Imbalance ด้วย Oversampling
# ══════════════════════════════════════════════
def oversample(
    class_images: dict[str, list[Path]],
    strategy: str = "max",
) -> dict[str, list[Path]]:
    """
    Oversampling เพื่อทำให้ทุก class มีจำนวนเท่ากัน
    strategy:
      - "max": oversample ทุก class ให้มีจำนวนเท่ากับ class ที่มีมากที่สุด
    """
    print(f"\n{'='*60}")
    print(f"  STEP 3: Oversampling (strategy={strategy})")
    print(f"{'='*60}")

    counts = {label: len(paths) for label, paths in class_images.items()}
    target_count = max(counts.values())

    print(f"  Before oversampling:")
    for label in sorted(counts):
        print(f"    {label:<25s}  {counts[label]:>6,}")
    print(f"  Target count per class: {target_count:,}")

    oversampled: dict[str, list[Path]] = {}

    for label, paths in class_images.items():
        current_count = len(paths)
        if current_count >= target_count:
            oversampled[label] = paths[:target_count]
        else:
            # สุ่มเพิ่มจากรูปที่มีอยู่
            additional_needed = target_count - current_count
            extra = random.choices(paths, k=additional_needed)
            oversampled[label] = paths + extra
        print(f"    {label}: {current_count:,} → {len(oversampled[label]):,} "
              f"(+{len(oversampled[label]) - current_count:,})")

    total_after = sum(len(v) for v in oversampled.values())
    print(f"\n  → Total after oversampling: {total_after:,}")
    return oversampled


# ══════════════════════════════════════════════
# STEP 4: แปลง Format (.jpg) และ Color Space (RGB)
# ══════════════════════════════════════════════
def convert_and_save(
    class_images: dict[str, list[Path]],
    output_dir: Path,
    quality: int = 95,
) -> dict[str, list[Path]]:
    """
    แปลงทุกภาพเป็น .jpg / RGB แล้วบันทึกลง output_dir
    โครงสร้างผลลัพธ์: output_dir / <label> / <filename>.jpg
    """
    print(f"\n{'='*60}")
    print(f"  STEP 4: Convert to .jpg / RGB")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, list[Path]] = {}
    total_converted = 0
    total_already_rgb = 0
    total_color_converted = 0

    for label, paths in class_images.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for idx, p in enumerate(paths):
            try:
                img = Image.open(p)

                # แปลง color space → RGB
                if img.mode == "RGB":
                    total_already_rgb += 1
                else:
                    # จัดการ RGBA, P, L, CMYK, etc.
                    if img.mode == "RGBA":
                        # สร้างพื้นหลังขาวสำหรับภาพโปร่งใส
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode == "P":
                        img = img.convert("RGBA")
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
                        img = background
                    else:
                        img = img.convert("RGB")
                    total_color_converted += 1

                # บันทึกเป็น .jpg
                out_name = f"{label}_{idx:05d}.jpg"
                out_path = label_dir / out_name
                img.save(out_path, format="JPEG", quality=quality)
                saved_paths.append(out_path)
                total_converted += 1

            except Exception as e:
                print(f"    ✗ Failed to convert: {p.name} ({e})")

        saved[label] = saved_paths
        print(f"    [{label}] saved {len(saved_paths):,} images")

    print(f"\n  → Total saved: {total_converted:,}")
    print(f"  → Already RGB: {total_already_rgb:,}")
    print(f"  → Color-converted: {total_color_converted:,}")
    return saved


# ══════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DOG EMOTION DATASET — Preprocessing Pipeline")
    print("=" * 60)

    # 0. ดาวน์โหลด / หา path ของ dataset อัตโนมัติ
    dataset_path = download_dataset()

    # สแกนหารูปภาพ
    class_images = scan_images(dataset_path)
    if not class_images:
        print("[ERROR] ไม่พบรูปภาพในโฟลเดอร์ dataset")
        sys.exit(1)

    total = sum(len(v) for v in class_images.values())
    print(f"\n[INFO] Found {total:,} images in {len(class_images)} classes")
    for label in sorted(class_images):
        print(f"  {label}: {len(class_images[label]):,}")

    # Pipeline
    class_images = remove_corrupted(class_images)     # Step 1
    class_images = remove_duplicates(class_images)     # Step 2
    class_images = oversample(class_images)            # Step 3
    saved = convert_and_save(class_images, OUTPUT_DIR) # Step 4

    # ── สรุปผล ──
    print(f"\n{'='*60}")
    print(f"  ✅ PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Classes: {len(saved)}")
    for label in sorted(saved):
        print(f"    {label:<25s}  {len(saved[label]):>6,} images")
    total_final = sum(len(v) for v in saved.values())
    print(f"  Total images: {total_final:,}")
    print(f"  Format: .jpg / RGB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
