# สรุปผลการดำเนินงาน: Branch `feature/preprocessing`

## 🎯 วัตถุประสงค์
จัดทำ Pipeline สำหรับการ Preprocessing ข้อมูลดิบ และประมวลผลรูปภาพ (Image Processing) ครบวงจร

## 🛠️ รายละเอียดโค้ด
### 1. `src/preprocessing.py`
- ค้นหาและดึง Dataset จาก Kaggle ผ่าน `kagglehub` อัตโนมัติ
- ตรวจสอบและลบไฟล์ภาพที่เสียหาย (Corrupted Images)
- ตรวจจับและกำจัดภาพซ้ำ (Duplicate Images) ด้วยวิธี Perceptual Hash (pHash)
- จัดการปัญหา Class Imbalance ด้วยเทคนิค Oversampling
- แปลง Format เป็น `.jpg` และ Color Space เป็น `RGB` บันทึกลงโฟลเดอร์ `data/processed/`

### 2. `src/image_processing.py`
- ประมวลผลกับทุกรูปภาพใน Dataset และบันทึกลง `data/image_processed/`
- **Resize:** ปรับขนาดภาพเป็น $233 \times 233$ ด้วย Lanczos Resampling
- **Normalization / Standardization:** ปรับสเกล pixel ช่วง $[0.0, 1.0]$ และ Z-score
- **Denoising:** กำจัด Salt & Pepper noise ด้วย Median Filter ($3 \times 3$)
- **Data Augmentation:** สร้างการแปลงภาพ (Flip, Rotation, Brightness, Contrast, Color Jitter)
- **Before & After Visualizations:** สร้างภาพเปรียบเทียบ Before & After (1 ภาพตัวอย่าง) ลงในโฟลเดอร์ `reports/`

## 📦 Dependencies
- `kagglehub`, `numpy`, `matplotlib`, `Pillow`
