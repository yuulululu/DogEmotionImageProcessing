# สรุปผลการดำเนินงาน: Branch `feature/datacollection`

## 🎯 วัตถุประสงค์
จัดเตรียมโครงสร้างโปรเจกต์และเขียนสคริปต์สำหรับดาวน์โหลดชุดข้อมูล Dog Emotion Dataset จาก Kaggle ผ่าน `kagglehub`

## 📁 โครงสร้างโฟลเดอร์ที่จัดเตรียม
- `src/` : ที่เก็บโค้ดโปรแกรมหลัก
- `reports/` : โฟลเดอร์สำหรับรายงานผลและรูปภาพวิเคราะห์
- `slides/` : โฟลเดอร์สำหรับเก็บสไลด์นำเสนอ

## 🛠️ รายละเอียดโค้ด `src/data_collection.py`
- ใช้ `kagglehub.dataset_download("mohitagarwal17/dog-emotion-datasetcleaned-version")` ในการดึงข้อมูลอัตโนมัติ
- คืนค่า Path ที่เก็บไฟล์ข้อมูลในเครื่องพร้อมแสดงผลเพื่อนำไปใช้ในขั้นตอนถัดไป

## 📦 Dependencies
- `kagglehub`
