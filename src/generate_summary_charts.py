"""
generate_summary_charts.py - สร้างกราฟสถิติแสดง Data Flow, Data Loss และการเปลี่ยนแปลงใน Pipeline
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_pipeline_loss_chart():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Chart 1: Data Flow & Retention / Loss Waterfall ---
    stages = [
        "1. Raw Dataset\n(Kaggle Source)",
        "2. Corrupted\nFiltered",
        "3. pHash Deduplication\n(Loss: Duplicates)",
        "4. Unique Cleaned\nBase Data",
        "5. Oversampling\n(Balanced Classes)",
        "6. Final Processed\n(with Augmentation)"
    ]
    
    # ตัวเลขสถิติภาพรวม
    values = [4000, 3995, 3850, 3850, 4800, 24000]
    changes = [0, -5, -145, 0, +950, +19200]
    colors = ["#3498db", "#e74c3c", "#e67e22", "#2ecc71", "#9b59b6", "#1abc9c"]

    bars = axes[0].bar(stages, values, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
    axes[0].set_title("Data Volume & Transition across Pipeline Stages", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val, chg in zip(bars, values, changes):
        height = bar.get_height()
        chg_text = f" ({chg:+,})" if chg != 0 else ""
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + max(values) * 0.02,
            f"{val:,}{chg_text}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    axes[0].tick_params(axis="x", labelsize=9)

    # --- Chart 2: Data Loss / Quality Breakdown in Preprocessing ---
    labels = ["Valid Clean\nImages (96.25%)", "Duplicate Images\nRemoved (3.63%)", "Corrupted / Broken\nRemoved (0.12%)"]
    sizes = [3850, 145, 5]
    colors_pie = ["#2ecc71", "#e67e22", "#e74c3c"]
    explode = (0.05, 0.1, 0.15)

    wedges, texts, autotexts = axes[1].pie(
        sizes, labels=labels, autopct="%1.2f%%", startangle=140,
        colors=colors_pie, explode=explode, shadow=True,
        textprops=dict(fontweight="bold", fontsize=10)
    )
    axes[1].set_title("Preprocessing Data Loss & Quality Breakdown", fontsize=13, fontweight="bold", pad=12)

    plt.suptitle("Dog Emotion Image Processing: Data Loss, Filtering & Pipeline Analysis", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    
    out_path = REPORTS_DIR / "data_loss_and_flow_analysis.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")

if __name__ == "__main__":
    generate_pipeline_loss_chart()
