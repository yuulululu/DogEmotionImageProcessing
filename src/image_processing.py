"""
image_processing.py - Image Processing Pipeline สำหรับ Dog Emotion Dataset
============================================================================
ฟังก์ชันและกระบวนการหลัก:
  1. ดึงรูปภาพจากโฟลเดอร์ผลลัพธ์ของ Preprocessing (data/processed/)
  2. ทำ Image Processing กับ "ทุกภาพใน Dataset":
     - Resize ให้มีขนาดเท่ากัน 233x233 (Lanczos)
     - Denoising กำจัด Salt & Pepper noise ด้วย Median Filter (3x3)
     - Normalization / Standardization
     - Data Augmentation (Flip, Rotation, Brightness, Contrast, Color Jitter)
     - บันทึกผลลัพธ์ทั้งหมดลงในโฟลเดอร์ data/image_processed/ แยกตาม class
  3. สร้างภาพเปรียบเทียบ Before & After "เพียง 1 ภาพตัวอย่าง" และเซฟเป็น .png ลงใน reports/
"""

import os
import sys
import shutil
import random
from pathlib import Path

import kagglehub
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# ──────────────────────────────────────────────
# CONFIG & PATHS
# ──────────────────────────────────────────────
TARGET_SIZE = (233, 233)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DATA_DIR = PROJECT_ROOT / "data" / "image_processed"

# เพิ่ม src ลงใน sys.path
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)
random.seed(42)


# ══════════════════════════════════════════════
# HELPER: ค้นหาและเตรียมข้อมูลจาก Preprocessing
# ══════════════════════════════════════════════
def get_processed_dataset(processed_dir: Path = PROCESSED_DATA_DIR) -> dict[str, list[Path]]:
    """
    สแกนหารูปภาพทั้งหมดในโฟลเดอร์ผลลัพธ์ของ Preprocessing (data/processed/)
    คืนค่าเป็น dict: { class_name: [image_paths...] }
    """
    # 1. หากยังไม่มีข้อมูล ให้รัน preprocessing.py อัตโนมัติ
    if not processed_dir.exists() or not any(processed_dir.rglob("*.jpg")):
        print(f"[INFO] ไม่พบรูปภาพใน {processed_dir.relative_to(PROJECT_ROOT)}")
        print(f"[INFO] กำลังรัน preprocessing pipeline เพื่อเตรียมข้อมูล...")
        try:
            import preprocessing
            preprocessing.main()
        except Exception as e:
            print(f"[WARN] เกิดข้อผิดพลาดขณะรัน preprocessing: {e}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dataset: dict[str, list[Path]] = {}

    if processed_dir.exists():
        for item in sorted(processed_dir.iterdir()):
            if item.is_dir():
                imgs = [p for p in sorted(item.glob("*")) if p.suffix.lower() in valid_exts]
                if imgs:
                    dataset[item.name] = imgs

    return dataset


# ══════════════════════════════════════════════
# 1. RESIZE (233x233)
# ══════════════════════════════════════════════
def resize_image(img: Image.Image, size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """ปรับขนาดภาพเป็น 233x233 พิกเซล ด้วย Lanczos Resampling คุณภาพสูง"""
    return img.resize(size, Image.Resampling.LANCZOS)


# ══════════════════════════════════════════════
# 2. NORMALIZATION & STANDARDIZATION
# ══════════════════════════════════════════════
def normalize_min_max(img_arr: np.ndarray) -> np.ndarray:
    """Min-Max Normalization: ปรับสเกล pixel ให้อยู่ในช่วง [0.0, 1.0]"""
    arr_float = img_arr.astype(np.float32)
    min_val = arr_float.min()
    max_val = arr_float.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr_float)
    return (arr_float - min_val) / (max_val - min_val)


def standardize_zscore(img_arr: np.ndarray) -> np.ndarray:
    """Z-score Standardization: ปรับให้ mean = 0 และ std = 1"""
    arr_float = img_arr.astype(np.float32)
    mean = np.mean(arr_float, axis=(0, 1), keepdims=True)
    std = np.std(arr_float, axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (arr_float - mean) / std


def standardize_imagenet(img_arr: np.ndarray) -> np.ndarray:
    """ImageNet Standardization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]"""
    norm = img_arr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (norm - mean) / std


# ══════════════════════════════════════════════
# 3. DENOISING (Salt & Pepper Noise Removal)
# ══════════════════════════════════════════════
def add_salt_and_pepper_noise(img: Image.Image, amount: float = 0.05, s_vs_p: float = 0.5) -> Image.Image:
    """จำลองการเพิ่ม Salt & Pepper noise ลงในรูปภาพเพื่อทดสอบ"""
    arr = np.array(img).copy()
    h, w, c = arr.shape
    num_noise = int(amount * h * w)

    # Salt
    num_salt = int(num_noise * s_vs_p)
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    arr[coords] = 255

    # Pepper
    num_pepper = int(num_noise * (1.0 - s_vs_p))
    coords = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
    arr[coords] = 0

    return Image.fromarray(arr)


def denoise_salt_and_pepper(img: Image.Image, filter_size: int = 3) -> Image.Image:
    """กำจัด Salt & Pepper Noise ด้วย Median Filtering (3x3)"""
    return img.filter(ImageFilter.MedianFilter(size=filter_size))


# ══════════════════════════════════════════════
# 4. DATA AUGMENTATION
# ══════════════════════════════════════════════
def apply_augmentations(img: Image.Image) -> dict[str, Image.Image]:
    """สร้างรูปภาพ Augmented หลากหลายรูปแบบ (Flip, Rotation, Brightness, Contrast, Color)"""
    flipped = ImageOps.mirror(img)
    rotated = img.rotate(15, resample=Image.Resampling.BICUBIC, expand=False)
    bright = ImageEnhance.Brightness(img).enhance(1.3)
    contrast = ImageEnhance.Contrast(img).enhance(1.4)
    color = ImageEnhance.Color(img).enhance(1.5)

    return {
        "Original": img,
        "Horizontal Flip": flipped,
        "Rotation (15°)": rotated,
        "Brightness (+30%)": bright,
        "Contrast (+40%)": contrast,
        "Color Jitter": color,
    }


# ══════════════════════════════════════════════
# 5. BEFORE & AFTER VISUALIZATIONS (สำหรับ 1 ภาพตัวอย่าง)
# ══════════════════════════════════════════════
def generate_before_after_reports(sample_img: Image.Image):
    """สร้างและบันทึกภาพเปรียบเทียบ Before & After สำหรับ 1 ภาพตัวอย่าง ลงใน reports/"""
    print(f"\n{'='*60}")
    print("  สร้างภาพสรุป Before & After (1 ภาพตัวอย่าง) → reports/")
    print(f"{'='*60}")

    # 1. Resize
    resized_img = resize_image(sample_img, TARGET_SIZE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample_img)
    axes[0].set_title(f"Before Resize\nSize: {sample_img.size[0]}x{sample_img.size[1]}", fontsize=11, fontweight="bold")
    axes[1].imshow(resized_img)
    axes[1].set_title(f"After Resize (Target)\nSize: {resized_img.size[0]}x{resized_img.size[1]}", fontsize=11, fontweight="bold")
    plt.suptitle("1. Image Resizing (233x233)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "before_after_resize.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: reports/before_after_resize.png")

    # 2. Normalization & Standardization
    arr = np.array(resized_img)
    norm = normalize_min_max(arr)
    std_z = standardize_zscore(arr)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(arr)
    axes[0, 0].set_title("Original (0-255)", fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(norm)
    axes[0, 1].set_title("Normalized [0.0, 1.0]", fontweight="bold")
    axes[0, 1].axis("off")
    std_vis = (std_z - std_z.min()) / (std_z.max() - std_z.min())
    axes[0, 2].imshow(std_vis)
    axes[0, 2].set_title("Standardized (Z-Score)", fontweight="bold")
    axes[0, 2].axis("off")
    axes[1, 0].hist(arr.ravel(), bins=50, color="gray", alpha=0.7)
    axes[1, 0].set_title(f"Raw Pixels (min={arr.min()}, max={arr.max()})")
    axes[1, 1].hist(norm.ravel(), bins=50, color="teal", alpha=0.7)
    axes[1, 1].set_title(f"Normalized (min={norm.min():.2f}, max={norm.max():.2f})")
    axes[1, 2].hist(std_z.ravel(), bins=50, color="crimson", alpha=0.7)
    axes[1, 2].set_title(f"Z-Score (mean={std_z.mean():.2f}, std={std_z.std():.2f})")
    plt.suptitle("2. Normalization & Standardization Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "before_after_normalization.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: reports/before_after_normalization.png")

    # 3. Denoising
    noisy_img = add_salt_and_pepper_noise(resized_img, amount=0.08)
    denoised_img = denoise_salt_and_pepper(noisy_img, filter_size=3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(resized_img)
    axes[0].set_title("Clean Image", fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(noisy_img)
    axes[1].set_title("Before: Salt & Pepper Noise (8%)", fontweight="bold", color="red")
    axes[1].axis("off")
    axes[2].imshow(denoised_img)
    axes[2].set_title("After: Median Filter Denoising (3x3)", fontweight="bold", color="green")
    axes[2].axis("off")
    plt.suptitle("3. Denoising (Salt & Pepper Noise Removal)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "before_after_denoising.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: reports/before_after_denoising.png")

    # 4. Data Augmentation
    aug_dict = apply_augmentations(resized_img)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for idx, (title, aug_img) in enumerate(aug_dict.items()):
        axes[idx].imshow(aug_img)
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].axis("off")
    plt.suptitle("4. Data Augmentation Transformations", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "before_after_augmentation.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: reports/before_after_augmentation.png")

    # 5. Full Pipeline Summary
    noisy_sample = add_salt_and_pepper_noise(resized_img, amount=0.05)
    denoised_sample = denoise_salt_and_pepper(noisy_sample)
    norm_sample = normalize_min_max(np.array(denoised_sample))
    aug_sample = ImageOps.mirror(denoised_sample)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    titles = [
        f"1. Raw Input\n({sample_img.size[0]}x{sample_img.size[1]})",
        f"2. Resized\n(233x233)",
        "3. Denoised\n(Median 3x3)",
        "4. Normalized\n[0.0, 1.0]",
        "5. Augmented\n(Horiz. Flip)",
    ]
    for ax, title, im in zip(axes, titles, [sample_img, resized_img, denoised_sample, norm_sample, aug_sample]):
        ax.imshow(im)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Complete Image Processing Pipeline Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "image_processing_pipeline_summary.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: reports/image_processing_pipeline_summary.png")


# ══════════════════════════════════════════════
# 6. BATCH PROCESSING — ทำกับทุกภาพใน Dataset
# ══════════════════════════════════════════════
def process_all_images(dataset: dict[str, list[Path]], output_dir: Path = FINAL_DATA_DIR):
    """
    ประมวลผลทุกรูปภาพใน Dataset:
      - Resize เป็น 233x233
      - Denoise (Median filter)
      - ทำ Data Augmentation
      - บันทึกผลลัพธ์ลงในโฟลเดอร์ data/image_processed/<class_name>/
    """
    print(f"\n{'='*60}")
    print(f"  ประมวลผลทุกภาพใน Dataset → บันทึกลง: {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"{'='*60}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_inputs = sum(len(paths) for paths in dataset.values())
    total_saved = 0

    print(f"  จำนวนภาพทั้งหมดที่จะประมวลผล: {total_inputs:,} ภาพ จาก {len(dataset)} คลาส\n")

    for class_name, img_paths in sorted(dataset.items()):
        class_out_dir = output_dir / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        class_saved_count = 0

        print(f"  ▶ กำลังประมวลผลคลาส [{class_name}] ({len(img_paths):,} ภาพ)...")

        for idx, img_path in enumerate(img_paths):
            try:
                img = Image.open(img_path).convert("RGB")

                # 1. Resize (233x233)
                resized = resize_image(img, TARGET_SIZE)

                # 2. Denoising
                denoised = denoise_salt_and_pepper(resized, filter_size=3)

                # บันทึกภาพหลักที่ผ่าน Resize + Denoise
                base_name = f"{class_name}_{idx:05d}"
                main_out_path = class_out_dir / f"{base_name}.jpg"
                denoised.save(main_out_path, format="JPEG", quality=95)
                class_saved_count += 1

                # 3. Data Augmentation (สร้างภาพ Augmented เสริม)
                aug_dict = apply_augmentations(denoised)
                for aug_key, aug_img in aug_dict.items():
                    if aug_key == "Original":
                        continue
                    aug_tag = aug_key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("%", "").replace("°", "")
                    aug_out_path = class_out_dir / f"{base_name}_{aug_tag}.jpg"
                    aug_img.save(aug_out_path, format="JPEG", quality=95)
                    class_saved_count += 1

            except Exception as e:
                print(f"    ✗ เกิดข้อผิดพลาดกับภาพ {img_path.name}: {e}")

        total_saved += class_saved_count
        print(f"    ✓ คลาส [{class_name}] บันทึกสำเร็จ: {class_saved_count:,} ภาพ (รวม Augmentations)")

    print(f"\n  → รวมภาพผลลัพธ์ทั้งหมดที่บันทึกลง {output_dir.relative_to(PROJECT_ROOT)}: {total_saved:,} ภาพ")
    return total_saved


# ══════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  IMAGE PROCESSING PIPELINE — Dog Emotion Dataset")
    print("=" * 60)

    # 1. โหลด dataset จากโฟลเดอร์ผลลัพธ์ preprocessing (data/processed/)
    dataset = get_processed_dataset(PROCESSED_DATA_DIR)

    if not dataset:
        print("[ERROR] ไม่พบข้อมูลภาพสำหรับประมวลผล")
        sys.exit(1)

    # 2. สุ่มเลือก 1 ภาพตัวอย่างสำหรับสร้าง Before & After ใน reports/
    all_sample_paths = [p for paths in dataset.values() for p in paths]
    sample_path = random.choice(all_sample_paths)
    sample_img = Image.open(sample_path).convert("RGB")
    print(f"[INFO] เลือกภาพตัวอย่างสำหรับ Before & After: {sample_path.name}")

    # สร้าง Before & After ลง reports/ เพียง 1 ภาพ
    generate_before_after_reports(sample_img)

    # 3. ประมวลผลกับทุกภาพใน Dataset และบันทึกลง data/image_processed/
    total_saved = process_all_images(dataset, FINAL_DATA_DIR)

    print(f"\n{'='*60}")
    print("  ✅ เสร็จสิ้นกระบวนการ Image Processing ทั้งหมด!")
    print(f"  • ภาพผลลัพธ์ทั้งหมดถูกบันทึกที่: {FINAL_DATA_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  • ภาพเปรียบเทียบ Before & After (1 ภาพ) ถูกบันทึกที่: {REPORTS_DIR.relative_to(PROJECT_ROOT)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
