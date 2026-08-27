# สรุปผลการดำเนินงาน: Branch `feature/data_splitting`

## 🎯 วัตถุประสงค์
ทำการแบ่งชุดข้อมูล (Data Splitting) ออกเป็น Train / Validation / Test สำหรับเตรียมนำไปเทรนโมเดล Machine Learning / Deep Learning

## 🛠️ รายละเอียดโค้ด `src/data_splitting.py`
1. **อัตราส่วนการแบ่ง 70% / 15% / 15%:**
   - `Train Set`: 70%
   - `Validation Set`: 15%
   - `Test Set`: 15%
2. **Stratified Split:** รักษาสัดส่วนการกระจายตัวของคลาสให้เท่าเทียมกันทุก Subset
3. **Group-aware Splitting & Reproducibility:**
   - กำหนด `Random Seed = 42` เพื่อให้ผลลัพธ์ Reproduce ได้
   - ใช้ `group_id` ป้องกัน Data Leakage ระหว่าง Train/Val/Test
4. **Copy ภาพไปยังโฟลเดอร์ผลลัพธ์:**
   - คัดลอกภาพแยกตามชุดข้อมูลจริงไปไว้ที่:
     - `data/splits/train/<class_name>/`
     - `data/splits/val/<class_name>/`
     - `data/splits/test/<class_name>/`
5. **บันทึก Manifest (.csv):**
   - บันทึก `dataset_manifest.csv`, `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv` ลงใน `reports/` และ `data/splits/`
6. **สร้างชาร์ตสรุปผล:**
   - บันทึก `reports/split_distribution.png`

## 📦 Dependencies
- `numpy`, `matplotlib`
