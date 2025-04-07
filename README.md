# 🐔🐄🐑🐎 Классификация животных по изображениям

## 📝 Описание проекта

Этот проект представляет собой систему классификации изображений с использованием модели глубокого обучения, позволяющей отличать **курицу**, **корову**, **овцу** и **лошадь**. Веб-интерфейс реализован с помощью **Streamlit**, серверная часть — на **FastAPI**. Приложение позволяет загружать изображение или рисовать его прямо в браузере, после чего оно отправляется на сервер, где выполняется предобработка и предсказание класса.

В качестве данных использован набор изображений, содержащий фотографии указанных животных ([Датасет на Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)).

### 🧠 Особенности:
- Поддержка загрузки и рисования изображений.
- Отображение предсказанного класса и вероятностей по всем категориям.
- Интерактивная визуализация.

---

## 📦 Используемые технологии

- Python 🐍
- TensorFlow / Keras 🤖
- FastAPI ⚡
- Streamlit 🌐
- OpenCV 📷
- NumPy 📊
- Pillow 🖼️

---

## 🔍 Сравниваемые модели

| Имя модели               | Точность | Полнота |      Precision      | F1-мера|
|--------------------------|----------|---------|---------------------|--------|
| **deep_cnn(work_№3)**    |  0.9181  | 0.9181  | 0.9182              | 0.9180 |
| **residual_cnn(work_№4)**|  0.8854  | 0.8854  | 0.8860              | 0.8852 |
| **simple_cnn(work_№3)**  |  0.8729  | 0.8729  | 0.8743              | 0.8728 |
| **simple_dnn(work_№2)**  |  0.5458  | 0.5458  | 0.5409              | 0.5378 |

**Лучшая модель по F1-мере: deep_cnn(work_№3)**

<p align="center"> <img src="https://i.imgur.com/VvGk8E3.png" alt="confusion_matrix" width="1000"/> </p>

---

## 📊 Визуализация результатов

Пример интерфейса StreamLit:

<p align="center"> <img src="https://i.imgur.com/fkbFT5t.png" alt="confusion_matrix" width="600"/> </p>

---

## 🚀 Развёртывание проекта локально

1. **Клонируйте репозиторий:**

```bash
git clone https://github.com/Poziloi/chicken_cow_horse_sheep-classification.git
cd chicken_cow_horse_sheep-classification
```

2. **Создайте виртуальное окружение и активируйте его:**

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. **Установите зависимости:**

```bash
pip install -r requirements.txt
```

4. **Запустите FastAPI-сервер:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. **Запустите Streamlit-приложение:**

```bash
streamlit run app.py
```

6. **Откройте приложение в браузере:**
[http://localhost:8501](http://localhost:8501)
(порт 8501 установлен по умолчанию)

---

## 🔗 Развёрнутые приложения

- 🎯 **Streamlit-интерфейс**: [https://chicken-cow-horse-sheep-classification.streamlit.app/](https://chicken-cow-horse-sheep-classification.streamlit.app/)
- ⚡ **API сервер (FastAPI)**: [https://chicken-cow-horse-sheep-classification.onrender.com/docs](https://chicken-cow-horse-sheep-classification.onrender.com/docs)

---

## 🧪 Примеры использования API

### Пример POST-запроса с изображением

```bash
curl -X POST "https://chicken-api.onrender.com/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.jpg"
```

### Ответ:

```json
{
  "predicted_class": "3",
  "probabilities": {
    "0": 1.52e-8,
    "1": 0.207,
    "2": 0.000045,
    "3": 0.793
  }
}
```

**Где:**
- `"0"` — курица 🐔  
- `"1"` — корова 🐄  
- `"2"` — овца 🐑  
- `"3"` — лошадь 🐎  

---
