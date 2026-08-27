# ภาพรวมโครงงาน Dog Emotion Image Processing

สรุปผลการทำงานของแต่ละโมดูลในโปรเจกต์:

## 1. Data Collection (`feature/datacollection`)
- สคริปต์: `src/data_collection.py`
- ดาวน์โหลด Dataset จาก Kaggle (`mohitagarwal17/dog-emotion-datasetcleaned-version`) อัตโนมัติด้วย `kagglehub`

## 2. Exploratory Data Analysis (`feature/eda`)
- สคริปต์: `src/eda.py`
- วิเคราะห์ Class Distribution, ขนาดภาพ (Width, Height, Aspect Ratio, File Size), RGB Color Histograms และตรวจหาภาพ Corrupted / Duplicates / Grayscale ปน
- บันทึกผลลัพธ์ลงใน `reports/figures/`

## 3. Preprocessing & Image Processing (`feature/preprocessing`)
- สคริปต์: `src/preprocessing.py`, `src/image_processing.py`
- Preprocessing: ลบ Corrupted Images, กำจัดภาพซ้ำด้วย pHash, ทำ Oversampling แก้ Class Imbalance, แปลงภาพเป็น `.jpg` / RGB
- Image Processing: Resize $233 \times 233$, Denoise ด้วย Median Filter ลบ Salt & Pepper noise, Data Augmentation, Normalization/Standardization

## 4. Data Splitting (`feature/data_splitting`)
- สคริปต์: `src/data_splitting.py`
- แบ่งข้อมูล Train / Validation / Test สัดส่วน 70% / 15% / 15% แบบ Stratified
- ระบบ Group-aware ป้องกัน Data Leakage และเซฟไฟล์ Manifest (.csv) พร้อม Copy ภาพแยกโฟลเดอร์ `data/splits/`
