"""
Streamlit web app สำหรับทดสอบโมเดล Image Classification (Ultralytics YOLO)
รันในเครื่อง:      streamlit run app.py
Deploy ฟรี:        push โฟลเดอร์นี้ขึ้น GitHub แล้วเชื่อมกับ https://share.streamlit.io
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AI Image Classifier", page_icon="🔮", layout="centered")

# ----------------------------------------------------------------------
# 1. โหลดโมเดล (cache ไว้ไม่ให้โหลดซ้ำทุกครั้งที่ interact กับหน้าเว็บ)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input(
    "Model path (.pt)",
    value="runs/classify/runs_classify/custom_classifier_exp/weights/best.pt",
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
st.title("🔮 AI Image Classification")
st.write("อัปโหลดภาพเพื่อให้โมเดลทำนายผล พร้อมแสดงเปอร์เซ็นต์ความมั่นใจของแต่ละคลาส")

uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    # 3. รัน inference
    with st.spinner("กำลังทำนายผล..."):
        results = model.predict(source=np.array(image), verbose=False)
        result = results[0]

    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    top1_name = result.names[top1_idx]

    with col2:
        st.metric("Top-1 Prediction", top1_name, f"{top1_conf*100:.2f}%")

    # 4. แสดง Top-5 เป็นกราฟแท่งแนวนอน (เรียงจากมากไปน้อย)
    st.subheader("📊 Top-5 ความน่าจะเป็น")

    top5_idx = result.probs.top5
    top5_conf = result.probs.top5conf.tolist()
    df_top5 = pd.DataFrame({
        "class": [result.names[i] for i in top5_idx],
        "percent": [c * 100 for c in top5_conf],
    })
    df_top5["label"] = df_top5["percent"].map(lambda p: f"{p:.1f}%")

    bar_chart = (
        alt.Chart(df_top5)
        .mark_bar(color="#22c55e", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("percent:Q", title="ความมั่นใจ (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("class:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("class:N", title="คลาส"), alt.Tooltip("percent:Q", title="%", format=".2f")],
        )
    )
    text_labels = bar_chart.mark_text(align="left", dx=5, color="white" if False else "black").encode(
        text="label:N"
    )
    st.altair_chart((bar_chart + text_labels).properties(height=35 * len(df_top5) + 20), use_container_width=True)

    with st.expander("ดูค่าตัวเลขแบบละเอียด"):
        for _, row in df_top5.iterrows():
            st.write(f"**{row['class']}**: {row['percent']:.2f}%")
else:
    st.info("👆 อัปโหลดภาพด้านบนเพื่อเริ่มทำนายผล")