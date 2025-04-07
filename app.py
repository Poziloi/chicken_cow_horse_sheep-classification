import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# URL API
API_URL = "https://chicken-cow-horse-sheep-classification.onrender.com/predict/"

# Список читаемых меток классов
CLASS_NAMES = ["chicken", "cow", "horse", "sheep"]

st.title("🐔🐄🐎🐑 Классификация изображений животных")

tab1, tab2 = st.tabs(["📷 Загрузить изображение", "✏️ Нарисовать изображение"])

image = None

with tab1:
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

with tab2:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=256,
        height=256,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data).astype("uint8")).convert("RGB")
        image = img

if image:
    st.image(image, caption="Входное изображение", use_container_width=True)

    if st.button("Классифицировать"):
        # Предобработка: ресайз и преобразование в байты
        img_resized = image.resize((64, 64))
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Отправка на API
        files = {"file": ("image.png", img_bytes, "image/png")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            # Получение метки класса
            predicted = result.get("predicted_class")
            if isinstance(predicted, int) and 0 <= predicted < len(CLASS_NAMES):
                predicted_label = CLASS_NAMES[predicted]
            else:
                predicted_label = predicted  # если API уже вернул строку

            st.subheader("✅ Предсказанный класс:")
            st.write(predicted_label)

            st.subheader("📊 Распределение вероятностей:")
            probs = result.get("probabilities", {})
            fig, ax = plt.subplots()
            ax.bar(probs.keys(), probs.values(), color="skyblue")
            ax.set_ylabel("Вероятность")
            ax.set_ylim([0, 1])
            st.pyplot(fig)
        else:
            st.error("Ошибка при обращении к API:")
            st.text(response.text)
