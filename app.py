"""
Streamlit web app สำหรับทดสอบโมเดล Image Classification (Ultralytics YOLO)
รันในเครื่อง:      streamlit run app.py
Deploy ฟรี:        push โฟลเดอร์นี้ขึ้น GitHub แล้วเชื่อมกับ https://share.streamlit.io
"""

import base64
import io

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Doggoo Emotional", layout="centered")

# สีไล่ตามอันดับ (rank 1 = เขียว, 2 = ส้ม, 3 = ม่วง, 4 = ฟ้า, 5 = ชมพู)
RANK_COLORS = [
    ("#22c55e", "#4ade80"),  # 1 green
    ("#f59e0b", "#fbbf24"),  # 2 orange
    ("#8b5cf6", "#a78bfa"),  # 3 purple
    ("#3b82f6", "#60a5fa"),  # 4 blue
    ("#ec4899", "#f472b6"),  # 5 pink
]

# ----------------------------------------------------------------------
# 1. โหลดโมเดล (cache ไว้ไม่ให้โหลดซ้ำทุกครั้งที่ interact กับหน้าเว็บ)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input(
    "Model path (.pt)",
    value="runs_classify/custom_classifier_exp/weights/best.pt",
    help="ใส่ path ของโมเดลที่เทรนเอง หรือใช้ yolo11n-cls.pt / yolov8n-cls.pt สำหรับทดสอบ",
)

try:
    model = load_model(model_path)
    st.sidebar.success(f"โหลดโมเดลสำเร็จ: {model_path}")
except Exception as e:
    st.sidebar.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ----------------------------------------------------------------------
# 2. หน้าหลัก — อัปโหลดภาพ
# ----------------------------------------------------------------------
st.title("Dog Emotion Classifier")
st.write("อัปโหลดภาพเพื่อให้โมเดลทำนายผล พร้อมแสดงเปอร์เซ็นต์ความมั่นใจของแต่ละคลาส")

uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # 3. รัน inference
    with st.spinner("กำลังทำนายผล..."):
        results = model.predict(source=np.array(image), verbose=False)
        result = results[0]

    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    top1_name = result.names[top1_idx]

    # เรียงทุกคลาสจากมากไปน้อย (โมเดลนี้มี 4 คลาส: angry/happy/relaxed/sad)
    n_classes = len(result.names)
    all_probs = sorted(
        [(result.names[i], float(result.probs.data[i])) for i in range(n_classes)],
        key=lambda x: x[1],
        reverse=True,
    )

    # ภาพเป็น base64 เพื่อฝังใน CSS background-image
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # แถวรายการ prediction แต่ละอันดับ
    # หมายเหตุ: เขียนทุกแถวเป็นบรรทัดเดียวไม่เว้นบรรทัดว่าง/ไม่เยื้อง เพื่อไม่ให้ตัวแปลง Markdown
    # ตีความเป็น code block (แม้จะ render ผ่าน st.html ที่ไม่ผ่าน Markdown แล้วก็ตาม แต่เขียนสะอาดไว้ก่อน)
    row_parts = []
    for rank, (cname, prob) in enumerate(all_probs, start=1):
        pct = prob * 100
        dark, light = RANK_COLORS[(rank - 1) % len(RANK_COLORS)]
        row_parts.append(
            f'<div class="pred-row">'
            f'<div class="pred-rank" style="background:{dark};">{rank}</div>'
            f'<div class="pred-name">{cname.capitalize()}</div>'
            f'<div class="pred-bar-bg">'
            f'<div class="pred-bar-fill" style="width:{max(pct, 2):.1f}%; background:linear-gradient(90deg,{light},{dark});"></div>'
            f'</div>'
            f'<div class="pred-pct">{pct:.2f}%</div>'
            f'</div>'
        )
    rows_html = "".join(row_parts)

    card_html = f"""
    <style>
        .pred-card {{
            display: flex;
            gap: 20px;
            background: #eef3ff;
            border-radius: 24px;
            padding: 20px;
            flex-wrap: wrap;
        }}
        .pred-photo {{
            position: relative;
            flex: 0 0 260px;
            height: 300px;
            border-radius: 18px;
            background-image: url("data:image/jpeg;base64,{img_b64}");
            background-size: cover;
            background-position: center;
            overflow: hidden;
        }}
        .pred-badge {{
            position: absolute;
            left: 12px;
            right: 12px;
            bottom: 12px;
            background: linear-gradient(90deg, #4ade80, #22c55e);
            border-radius: 999px;
            padding: 10px 14px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .pred-badge .label {{
            color: #ecfdf5;
            font-size: 12px;
            font-weight: 500;
        }}
        .pred-badge .value {{
            color: white;
            font-size: 20px;
            font-weight: 800;
        }}
        .pred-list {{
            flex: 1 1 300px;
            background: white;
            border-radius: 18px;
            padding: 20px 22px;
        }}
        .pred-list-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
            color: #1e3a8a;
            font-size: 17px;
            margin-bottom: 16px;
        }}
        .pred-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .pred-rank {{
            width: 26px;
            height: 26px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 12px;
            flex-shrink: 0;
        }}
        .pred-name {{
            width: 80px;
            font-weight: 600;
            color: #1e293b;
            flex-shrink: 0;
        }}
        .pred-bar-bg {{
            flex: 1;
            background: #e5e9f5;
            border-radius: 999px;
            height: 13px;
            overflow: hidden;
        }}
        .pred-bar-fill {{
            height: 100%;
            border-radius: 999px;
        }}
        .pred-pct {{
            width: 60px;
            text-align: right;
            font-weight: 700;
            color: #1e293b;
            flex-shrink: 0;
        }}
    </style>

    <div class="pred-card">
        <div class="pred-photo">
            <div class="pred-badge">
                <div class="label">Predicted Emotion</div>
                <div class="value">{top1_name.capitalize()}</div>
            </div>
        </div>
        <div class="pred-list">
            <div class="pred-list-header">📊 Predictions</div>
            {rows_html}
        </div>
    </div>
    """

    st.html(card_html)

    with st.expander("ดูค่าตัวเลขแบบละเอียด"):
        for cname, prob in all_probs:
            st.write(f"**{cname}**: {prob*100:.2f}%")
else:
    st.info("👆 อัปโหลดภาพด้านบนเพื่อเริ่มทำนายผล")
