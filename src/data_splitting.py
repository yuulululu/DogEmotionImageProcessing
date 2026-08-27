"""
data_splitting.py - Stratified Data Splitting Pipeline (Train / Val / Test)
============================================================================
ฟังก์ชันและกระบวนการหลัก:
  1. แบ่งชุดข้อมูล Train / Validation / Test ด้วยอัตราส่วน 70% / 15% / 15%
  2. ใช้ Stratified Split เพื่อรักษาสัดส่วนของ Class ให้เท่ากันทุก Subset
  3. กำหนด Random Seed (42) ให้ผลลัพธ์ Reproducible 100%
     พร้อมระบบ Group-aware Splitting ป้องกัน Data Leakage
  4. คัดลอก (Copy) ไฟล์ภาพที่แบ่งแล้วไปยังโฟลเดอร์แยก:
     - data/splits/train/<class_name>/
     - data/splits/val/<class_name>/
     - data/splits/test/<class_name>/
  5. บันทึก Manifest (.csv) รายละเอียดครบถ้วนลงใน reports/ และ data/splits/
  6. สร้างภาพสรุปและสถิติการกระจายตัวของข้อมูลในแต่ละ Split
"""

import os
import sys
import csv
import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIG & PATHS
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_PROCESSED_DIR = PROJECT_ROOT / "data" / "image_processed"
SPLITS_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# กำหนด Seed ให้คงที่
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════
# 1. HELPER: ค้นหาข้อมูลภาพและจัดกลุ่ม Group ID (ป้องกัน Leakage)
# ══════════════════════════════════════════════
def extract_group_id(filename: str) -> str:
    """
    ดึง Base Group ID เพื่อป้องกัน Data Leakage
    เช่น:
      'happy_00012.jpg'                  -> 'happy_00012'
      'happy_00012_horizontal_flip.jpg'  -> 'happy_00012'
      'happy_00012_rotation_15.jpg'      -> 'happy_00012'
    ภาพที่มาจากภาพต้นฉบับเดียวกันจะมี group_id เหมือนกันและจะถูกจัดเข้า Split เดียวกันเสมอ
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return stem


def compute_md5(filepath: Path) -> str:
    """คำนวณ MD5 hash ของไฟล์"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset_records(data_dir: Path) -> list[dict]:
    """สแกนหารูปภาพทั้งหมดและสร้าง record ข้อมูลสำหรับทำ Stratified Split"""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []

    if not data_dir.exists():
        return []

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in valid_exts:
            class_name = img_path.parent.name
            group_id = extract_group_id(img_path.name)
            file_size = img_path.stat().st_size

            records.append({
                "filepath": img_path,
                "relative_path": str(img_path.relative_to(PROJECT_ROOT)),
                "filename": img_path.name,
                "class_label": class_name,
                "group_id": group_id,
                "file_size_bytes": file_size,
            })
    return records


# ══════════════════════════════════════════════
# 2. STRATIFIED SPLIT 70 / 15 / 15 (Group-Aware & Reproducible)
# ══════════════════════════════════════════════
def stratified_group_split(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """ทำการแบ่งข้อมูลแบบ Stratified Split ร่วมกับ Group-aware"""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "อัตราส่วนรวมต้องเท่ากับ 1.0"
    rng = random.Random(seed)

    class_groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        class_groups[r["class_label"]][r["group_id"]].append(r)

    labeled_records = []

    print(f"\n{'='*65}")
    print(f"  STRATIFIED GROUP SPLIT ({int(train_ratio*100)}% / {int(val_ratio*100)}% / {int(test_ratio*100)}%) | Seed={seed}")
    print(f"{'='*65}")
    print(f"  {'Class':<20s} | {'Total Groups':<12s} | {'Train':<8s} | {'Val':<8s} | {'Test':<8s}")
    print(f"  {'-'*60}")

    for class_name, groups_dict in sorted(class_groups.items()):
        group_keys = list(groups_dict.keys())
        rng.shuffle(group_keys)

        n_total_groups = len(group_keys)
        n_train = int(round(n_total_groups * train_ratio))
        n_val = int(round(n_total_groups * val_ratio))
        n_test = n_total_groups - n_train - n_val

        train_keys = set(group_keys[:n_train])
        val_keys = set(group_keys[n_train:n_train + n_val])
        test_keys = set(group_keys[n_train + n_val:])

        print(f"  {class_name:<20s} | {n_total_groups:<12d} | {len(train_keys):<8d} | {len(val_keys):<8d} | {len(test_keys):<8d}")

        for gid, rec_list in groups_dict.items():
            if gid in train_keys:
                split_tag = "train"
            elif gid in val_keys:
                split_tag = "val"
            else:
                split_tag = "test"

            for rec in rec_list:
                rec_copy = rec.copy()
                rec_copy["split"] = split_tag
                labeled_records.append(rec_copy)

    return labeled_records


# ══════════════════════════════════════════════
# 3. ตรวจสอบ DATA LEAKAGE VERIFICATION
# ══════════════════════════════════════════════
def verify_no_data_leakage(records: list[dict]) -> bool:
    """ตรวจสอบและยืนยันว่าไม่มี Group ID เดียวกันหลุดข้าม Split"""
    split_groups = defaultdict(set)
    for r in records:
        split_groups[r["split"]].add(r["group_id"])

    train_val_overlap = split_groups["train"].intersection(split_groups["val"])
    train_test_overlap = split_groups["train"].intersection(split_groups["test"])
    val_test_overlap = split_groups["val"].intersection(split_groups["test"])

    print(f"\n{'='*65}")
    print("  DATA LEAKAGE AUDIT")
    print(f"{'='*65}")
    print(f"  • Train-Val overlap  : {len(train_val_overlap)} groups")
    print(f"  • Train-Test overlap : {len(train_test_overlap)} groups")
    print(f"  • Val-Test overlap   : {len(val_test_overlap)} groups")

    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("  ✅ PASS: ไม่พบการรั่วไหลของข้อมูลระหว่าง Split (Zero Data Leakage)")
        return True
    else:
        print("  ❌ FAIL: พบ Data Leakage ข้ามชุดข้อมูล!")
        return False


# ══════════════════════════════════════════════
# 4. คัดลอกภาพไปยังโฟลเดอร์แยกตาม Split (Copy to Split Folders)
# ══════════════════════════════════════════════
def copy_images_to_splits(records: list[dict], output_base_dir: Path = SPLITS_OUTPUT_DIR):
    """
    คัดลอกไฟล์รูปภาพไปยังโฟลเดอร์โครงสร้าง Train / Val / Test แยกตามแต่ละ Class
    โครงสร้าง:
      data/splits/train/<class_name>/<filename>
      data/splits/val/<class_name>/<filename>
      data/splits/test/<class_name>/<filename>
    """
    print(f"\n{'='*65}")
    print(f"  COPYING SPLIT IMAGES → {output_base_dir.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*65}")

    counts = defaultdict(int)
    for rec in records:
        src_path: Path = rec["filepath"]
        split_name = rec["split"]
        class_name = rec["class_label"]

        dest_dir = output_base_dir / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        shutil.copy2(src_path, dest_path)
        counts[split_name] += 1

    print(f"  ✓ Copied Train Images : {counts['train']:,} files")
    print(f"  ✓ Copied Val Images   : {counts['val']:,} files")
    print(f"  ✓ Copied Test Images  : {counts['test']:,} files")
    print(f"  → Total Copied: {sum(counts.values()):,} files")


# ══════════════════════════════════════════════
# 5. บันทึก MANIFEST (.CSV)
# ══════════════════════════════════════════════
def save_manifests(records: list[dict]):
    """บันทึก Manifest รายชื่อไฟล์ทั้งหมดและรายชื่อไฟล์แยกแต่ละ Split ลงเป็น .csv"""
    print(f"\n{'='*65}")
    print("  SAVING MANIFEST FILES (.CSV)")
    print(f"{'='*65}")

    fields = ["filename", "class_label", "split", "group_id", "relative_path", "file_size_bytes"]

    # 1. บันทึก Full Manifest
    full_manifest_path = REPORTS_DIR / "dataset_manifest.csv"
    with open(full_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"  → Saved Full Manifest: {full_manifest_path.relative_to(PROJECT_ROOT)} ({len(records):,} rows)")

    # 2. บันทึกแยก Train / Val / Test
    for split_name in ("train", "val", "test"):
        split_records = [r for r in records if r["split"] == split_name]
        
        # เซฟใน reports/
        split_csv_path = REPORTS_DIR / f"{split_name}_manifest.csv"
        with open(split_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_records)

        # เซฟสำเนาใน data/splits/
        data_split_csv = SPLITS_OUTPUT_DIR / f"{split_name}_manifest.csv"
        with open(data_split_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_records)

        print(f"  → Saved {split_name.upper():<5s} Manifest: {split_csv_path.relative_to(PROJECT_ROOT)} ({len(split_records):,} rows)")


# ══════════════════════════════════════════════
# 6. สรุปสถิติ & สร้างกราฟการกระจายตัวของ Split
# ══════════════════════════════════════════════
def generate_split_report_chart(records: list[dict]):
    """สร้างกราฟแสดงสัดส่วน Class ในแต่ละ Split เพื่อยืนยันคุณสมบัติ Stratified"""
    classes = sorted(list(set(r["class_label"] for r in records)))
    splits = ["train", "val", "test"]

    split_class_counts = {s: Counter(r["class_label"] for r in records if r["split"] == s) for s in splits}

    total_images = len(records)
    train_count = sum(split_class_counts["train"].values())
    val_count = sum(split_class_counts["val"].values())
    test_count = sum(split_class_counts["test"].values())

    print(f"\n{'='*65}")
    print("  FINAL SPLIT SUMMARY")
    print(f"{'='*65}")
    print(f"  Total Images : {total_images:,}")
    print(f"  Train Set    : {train_count:>6,} ({train_count/total_images*100:.1f}%)")
    print(f"  Val Set      : {val_count:>6,} ({val_count/total_images*100:.1f}%)")
    print(f"  Test Set     : {test_count:>6,} ({test_count/total_images*100:.1f}%)")
    print(f"{'='*65}")

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"train": "#2ecc71", "val": "#3498db", "test": "#e74c3c"}

    for i, s in enumerate(splits):
        counts = [split_class_counts[s][c] for c in classes]
        pcts = [counts[j] / (sum(split_class_counts[k][c] for k in splits) or 1) * 100 for j, c in enumerate(classes)]
        bars = ax.bar(x + (i - 1) * width, counts, width, label=f"{s.capitalize()} (n={sum(counts):,})", color=colors[s], edgecolor="black", linewidth=0.5)

        for bar, cnt, pct in zip(bars, counts, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{cnt}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Class Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=12, fontweight="bold")
    ax.set_title("Stratified Train / Validation / Test Split Distribution (70% / 15% / 15%)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = REPORTS_DIR / "split_distribution.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\n  → Saved Split Distribution Chart: {chart_path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  DATA SPLITTING PIPELINE — Dog Emotion Dataset")
    print("=" * 65)

    # 1. เลือกลำดับโฟลเดอร์ข้อมูลต้นทาง
    target_data_dir = None
    for candidate in [IMAGE_PROCESSED_DIR, PROCESSED_DATA_DIR]:
        if candidate.exists() and any(candidate.rglob("*.jpg")):
            target_data_dir = candidate
            break

    if target_data_dir is None:
        print("[INFO] ยังไม่พบโฟลเดอร์ภาพที่ประมวลผลแล้ว กำลังรัน preprocessing pipeline...")
        try:
            import preprocessing
            preprocessing.main()
            target_data_dir = PROCESSED_DATA_DIR
        except Exception as e:
            print(f"[ERROR] ไม่สามารถเตรียมข้อมูลได้: {e}")
            sys.exit(1)

    print(f"[INFO] ใช้ข้อมูลจาก: {target_data_dir.relative_to(PROJECT_ROOT)}")

    # 2. โหลด records
    records = load_dataset_records(target_data_dir)
    if not records:
        print("[ERROR] ไม่พบไฟล์ภาพในโฟลเดอร์เป้าหมาย")
        sys.exit(1)

    print(f"[INFO] สแกนพบรูปภาพทั้งหมด {len(records):,} ภาพ")
    print(f"[INFO] จำนวนคลาส: {len(set(r['class_label'] for r in records))}")

    # 3. Stratified Group Split (70/15/15)
    split_records = stratified_group_split(records, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)

    # 4. ตรวจสอบ Zero Data Leakage
    verify_no_data_leakage(split_records)

    # 5. คัดลอกภาพไปยังโฟลเดอร์แยก data/splits/train, val, test
    copy_images_to_splits(split_records, SPLITS_OUTPUT_DIR)

    # 6. บันทึกไฟล์ Manifests (.csv)
    save_manifests(split_records)

    # 7. สร้างรายงานและกราฟสรุป
    generate_split_report_chart(split_records)

    print(f"\n{'='*65}")
    print("  ✅ Data Splitting และ Copy ไฟล์เสร็จสมบูรณ์เรียบร้อย!")
    print("=" * 65)


if __name__ == "__main__":
    main()
