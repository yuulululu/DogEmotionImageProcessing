"""
eda.py - Exploratory Data Analysis (EDA) สำหรับ Dog Emotion Dataset
====================================================================
วิเคราะห์เชิงปริมาณและเชิงคุณภาพ:
  1. นับจำนวนภาพทั้งหมด แยกตาม class/label  → ตรวจ Class Imbalance
  2. การกระจายขนาดภาพ (width, height, aspect ratio, file size)
  3. Color Histogram (RGB channels)
  4. ตรวจไฟล์เสีย, รูปซ้ำ (duplicate), รูป grayscale ที่ปนอยู่

ผลลัพธ์ทั้งหมดจะบันทึกลงโฟลเดอร์  reports/figures/
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import kagglehub
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ใช้ backend ที่ไม่ต้องแสดงหน้าจอ
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────────
# 0. กำหนดค่าเริ่มต้น
# ──────────────────────────────────────────────
DATASET_SLUG = "mohitagarwal17/dog-emotion-datasetcleaned-version"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# โฟลเดอร์เก็บผลลัพธ์
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    """ดาวน์โหลด dataset ผ่าน kagglehub และคืน path ที่เก็บไฟล์"""
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"[INFO] Dataset path: {path}")
    return Path(path)


def scan_images(root: Path):
    """
    สแกนหารูปภาพทั้งหมดใต้ root อัตโนมัติ
    คืน list ของ dict ที่มี path, label (ชื่อโฟลเดอร์แม่)
    """
    records = []
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() in IMAGE_EXTENSIONS and img_path.is_file():
            label = img_path.parent.name  # ใช้ชื่อโฟลเดอร์แม่เป็น label
            records.append({"path": img_path, "label": label})
    return records


# ──────────────────────────────────────────────
# 1. นับจำนวนภาพ & Class Imbalance
# ──────────────────────────────────────────────
def analyze_class_distribution(records):
    """นับจำนวนภาพแยกตาม class/label และบันทึกกราฟ"""
    label_counts = Counter(r["label"] for r in records)
    labels = sorted(label_counts.keys())
    counts = [label_counts[l] for l in labels]

    total = sum(counts)
    print(f"\n{'='*60}")
    print(f"  1) CLASS DISTRIBUTION  (Total images: {total:,})")
    print(f"{'='*60}")
    for l, c in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = c / total * 100
        print(f"    {l:<25s}  {c:>6,}  ({pct:.1f}%)")

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Class / Label", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution (Class Imbalance Check)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    # แสดงจำนวนบน bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{count:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_path = FIGURES_DIR / "01_class_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {save_path}")
    return label_counts


# ──────────────────────────────────────────────
# 2. การกระจายขนาดภาพ
# ──────────────────────────────────────────────
def analyze_image_sizes(records):
    """วิเคราะห์ width, height, aspect ratio, file size"""
    widths, heights, ratios, file_sizes = [], [], [], []
    for r in records:
        try:
            img = Image.open(r["path"])
            w, h = img.size
            widths.append(w)
            heights.append(h)
            ratios.append(w / h if h > 0 else 0)
            file_sizes.append(r["path"].stat().st_size / 1024)  # KB
        except Exception:
            continue

    print(f"\n{'='*60}")
    print(f"  2) IMAGE SIZE DISTRIBUTION")
    print(f"{'='*60}")
    for name, data in [("Width (px)", widths), ("Height (px)", heights),
                        ("Aspect Ratio", ratios), ("File Size (KB)", file_sizes)]:
        arr = np.array(data)
        print(f"    {name:<20s}  min={arr.min():.1f}  max={arr.max():.1f}  "
              f"mean={arr.mean():.1f}  std={arr.std():.1f}")

    # ── 4-panel histogram ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(widths, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title("Width Distribution (px)")
    axes[0, 0].set_xlabel("Width")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(heights, bins=50, color="salmon", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title("Height Distribution (px)")
    axes[0, 1].set_xlabel("Height")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(ratios, bins=50, color="mediumseagreen", edgecolor="black", alpha=0.8)
    axes[1, 0].set_title("Aspect Ratio Distribution (W/H)")
    axes[1, 0].set_xlabel("Aspect Ratio")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(file_sizes, bins=50, color="orchid", edgecolor="black", alpha=0.8)
    axes[1, 1].set_title("File Size Distribution (KB)")
    axes[1, 1].set_xlabel("File Size (KB)")
    axes[1, 1].set_ylabel("Count")

    plt.suptitle("Image Size Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = FIGURES_DIR / "02_image_size_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {save_path}")

    return widths, heights, ratios, file_sizes


# ──────────────────────────────────────────────
# 3. Color Histogram (RGB Channels)
# ──────────────────────────────────────────────
def analyze_color_histogram(records, sample_size=500):
    """
    คำนวณ average color histogram จากตัวอย่างภาพ
    แสดง histogram แยก R, G, B channel
    """
    print(f"\n{'='*60}")
    print(f"  3) COLOR HISTOGRAM  (sample={sample_size})")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    indices = rng.choice(len(records), size=min(sample_size, len(records)), replace=False)

    hist_r = np.zeros(256, dtype=np.float64)
    hist_g = np.zeros(256, dtype=np.float64)
    hist_b = np.zeros(256, dtype=np.float64)
    count = 0

    for idx in indices:
        try:
            img = Image.open(records[idx]["path"]).convert("RGB")
            arr = np.array(img)
            hist_r += np.histogram(arr[:, :, 0], bins=256, range=(0, 256))[0]
            hist_g += np.histogram(arr[:, :, 1], bins=256, range=(0, 256))[0]
            hist_b += np.histogram(arr[:, :, 2], bins=256, range=(0, 256))[0]
            count += 1
        except Exception:
            continue

    if count > 0:
        hist_r /= count
        hist_g /= count
        hist_b /= count

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(256)
    ax.plot(x, hist_r, color="red", alpha=0.7, linewidth=1.2, label="Red")
    ax.plot(x, hist_g, color="green", alpha=0.7, linewidth=1.2, label="Green")
    ax.plot(x, hist_b, color="blue", alpha=0.7, linewidth=1.2, label="Blue")
    ax.fill_between(x, hist_r, alpha=0.15, color="red")
    ax.fill_between(x, hist_g, alpha=0.15, color="green")
    ax.fill_between(x, hist_b, alpha=0.15, color="blue")
    ax.set_xlabel("Pixel Intensity (0-255)", fontsize=12)
    ax.set_ylabel("Average Frequency", fontsize=12)
    ax.set_title(f"Average RGB Color Histogram (n={count})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 255)
    plt.tight_layout()

    save_path = FIGURES_DIR / "03_color_histogram.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → Sampled {count} images")
    print(f"  → Saved: {save_path}")

    # ── Per-class histogram ──
    labels = sorted(set(r["label"] for r in records))
    if len(labels) <= 10:
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
        for i, label in enumerate(labels):
            label_records = [r for r in records if r["label"] == label]
            sample_idx = rng.choice(len(label_records),
                                    size=min(100, len(label_records)), replace=False)
            h_r = np.zeros(256, dtype=np.float64)
            h_g = np.zeros(256, dtype=np.float64)
            h_b = np.zeros(256, dtype=np.float64)
            cnt = 0
            for si in sample_idx:
                try:
                    img = Image.open(label_records[si]["path"]).convert("RGB")
                    arr = np.array(img)
                    h_r += np.histogram(arr[:, :, 0], bins=256, range=(0, 256))[0]
                    h_g += np.histogram(arr[:, :, 1], bins=256, range=(0, 256))[0]
                    h_b += np.histogram(arr[:, :, 2], bins=256, range=(0, 256))[0]
                    cnt += 1
                except Exception:
                    continue
            if cnt > 0:
                h_r /= cnt
                h_g /= cnt
                h_b /= cnt
            ax = axes[0][i]
            ax.plot(x, h_r, color="red", alpha=0.7, linewidth=0.8)
            ax.plot(x, h_g, color="green", alpha=0.7, linewidth=0.8)
            ax.plot(x, h_b, color="blue", alpha=0.7, linewidth=0.8)
            ax.set_title(f"{label}\n(n={cnt})", fontsize=10, fontweight="bold")
            ax.set_xlim(0, 255)
            if i == 0:
                ax.set_ylabel("Avg Frequency")
        plt.suptitle("Per-Class RGB Histogram", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_path = FIGURES_DIR / "03_color_histogram_per_class.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved: {save_path}")


# ──────────────────────────────────────────────
# 4. ตรวจไฟล์เสีย / รูปซ้ำ / Grayscale ปน
# ──────────────────────────────────────────────
def check_quality_issues(records):
    """ตรวจหาปัญหาด้านคุณภาพของภาพ"""
    print(f"\n{'='*60}")
    print(f"  4) QUALITY ISSUES CHECK")
    print(f"{'='*60}")

    corrupted = []
    grayscale_images = []
    hash_map = defaultdict(list)  # md5 -> [paths]

    for r in records:
        fpath = r["path"]

        # 4a. ตรวจไฟล์เสีย
        try:
            img = Image.open(fpath)
            img.verify()  # ตรวจสอบความสมบูรณ์
        except Exception as e:
            corrupted.append((fpath, str(e)))
            continue

        # 4b. ตรวจรูปซ้ำ (hash-based)
        try:
            md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
            hash_map[md5].append(fpath)
        except Exception:
            pass

        # 4c. ตรวจ grayscale ที่ปนอยู่
        try:
            img = Image.open(fpath)
            if img.mode == "L":
                grayscale_images.append(fpath)
            elif img.mode == "RGB":
                arr = np.array(img)
                # ถ้า R == G == B ทุก pixel → เป็น grayscale
                if np.allclose(arr[:, :, 0], arr[:, :, 1]) and \
                   np.allclose(arr[:, :, 1], arr[:, :, 2]):
                    grayscale_images.append(fpath)
        except Exception:
            pass

    # ── แสดงผล Corrupted ──
    print(f"\n  4a) Corrupted Files: {len(corrupted)}")
    if corrupted:
        for fp, err in corrupted[:20]:
            print(f"      ✗ {fp}  ({err})")
        if len(corrupted) > 20:
            print(f"      ... and {len(corrupted) - 20} more")

    # ── แสดงผล Duplicates ──
    duplicate_groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    total_duplicates = sum(len(p) - 1 for p in duplicate_groups.values())
    print(f"\n  4b) Duplicate Images: {total_duplicates} duplicates in {len(duplicate_groups)} groups")
    if duplicate_groups:
        for i, (h, paths) in enumerate(list(duplicate_groups.items())[:10]):
            print(f"      Group {i+1} (hash={h[:12]}...): {len(paths)} copies")
            for p in paths[:3]:
                print(f"          - {p}")
            if len(paths) > 3:
                print(f"          ... and {len(paths) - 3} more")

    # ── แสดงผล Grayscale ──
    print(f"\n  4c) Grayscale Images Found: {len(grayscale_images)}")
    if grayscale_images:
        for fp in grayscale_images[:20]:
            print(f"      ◉ {fp}")
        if len(grayscale_images) > 20:
            print(f"      ... and {len(grayscale_images) - 20} more")

    # ── Summary chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    issue_labels = ["Corrupted\nFiles", "Duplicate\nImages", "Grayscale\nImages"]
    issue_counts = [len(corrupted), total_duplicates, len(grayscale_images)]
    bar_colors = ["#e74c3c", "#f39c12", "#3498db"]
    bars = ax.bar(issue_labels, issue_counts, color=bar_colors, edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, issue_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(issue_counts) * 0.02,
                f"{count:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Data Quality Issues Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = FIGURES_DIR / "04_quality_issues.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  → Saved: {save_path}")

    return corrupted, duplicate_groups, grayscale_images


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DOG EMOTION DATASET — Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # ดาวน์โหลด / หา path ของ dataset
    dataset_path = download_dataset()

    # สแกนหารูปภาพ
    records = scan_images(dataset_path)
    if not records:
        print("[ERROR] ไม่พบรูปภาพในโฟลเดอร์ dataset")
        sys.exit(1)

    print(f"\n[INFO] Found {len(records):,} images")
    print(f"[INFO] Labels: {sorted(set(r['label'] for r in records))}")
    print(f"[INFO] Output dir: {FIGURES_DIR}")

    # รันการวิเคราะห์ทั้งหมด
    analyze_class_distribution(records)
    analyze_image_sizes(records)
    analyze_color_histogram(records)
    corrupted, duplicates, grayscale = check_quality_issues(records)

    # ── สรุปผล ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"    Total images       : {len(records):,}")
    print(f"    Classes            : {len(set(r['label'] for r in records))}")
    print(f"    Corrupted files    : {len(corrupted)}")
    print(f"    Duplicate images   : {sum(len(p)-1 for p in duplicates.values())}")
    print(f"    Grayscale images   : {len(grayscale)}")
    print(f"    Figures saved to   : {FIGURES_DIR}")
    print(f"{'='*60}")
    print("  ✅ EDA completed successfully!")


if __name__ == "__main__":
    main()
