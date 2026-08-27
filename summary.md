# สรุปผลการดำเนินงาน: Branch `feature/eda`

## 🎯 วัตถุประสงค์
ทำการวิเคราะห์สำรวจข้อมูลรูปภาพ (Exploratory Data Analysis - EDA) ทั้งในเชิงปริมาณและเชิงคุณภาพ

## 🛠️ รายละเอียดโค้ด `src/eda.py`
1. **Class Distribution Analysis**: ตรวจสอบจำนวนภาพและสัดส่วนแต่ละ Label เพื่อเช็ค Class Imbalance (`01_class_distribution.png`)
2. **Image Size Distribution**: วิเคราะห์การกระจายตัวของ Width, Height, Aspect Ratio และ File Size (`02_image_size_distribution.png`)
3. **Color Histogram**: คำนวณ RGB Color Histogram ค่าเฉลี่ยทั้ง Dataset และแยกตามแต่ละ Class (`03_color_histogram.png`, `03_color_histogram_per_class.png`)
4. **Data Quality Issues Check**: ตรวจสอบไฟล์ภาพที่เสีย (Corrupted), ภาพซ้ำ (MD5 Hashing), และภาพที่เป็น Grayscale ที่ปนอยู่ (`04_quality_issues.png`)

## 📊 ผลลัพธ์ที่บันทึก
- รูปภาพกราฟและชาร์ตสรุปทั้งหมดจะถูกบันทึกอัตโนมัติลงในโฟลเดอร์ `reports/figures/`

## 📦 Dependencies
- `kagglehub`, `numpy`, `matplotlib`, `Pillow`
