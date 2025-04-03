# 🧠 Face Segmentation App

A web app that performs **face and hair segmentation** on images using deep learning. Upload an image containing exactly **one face**, and receive a transparent background version with only the segmented head/face area.

---

## 🚀 Features

- ✅ Validates that image has only **one human face**
- 🎯 Uses **BiSeNet** for high-quality face segmentation
- 🎨 Outputs RGBA image with a **transparent background**
- 📦 No need to store large models locally – downloads model from Google Drive
- 🌐 Deployed with [Streamlit](https://streamlit.io)
- ☁️ Runs on [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 💻 Setup Instructions

### 🔹 Create & activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate

macOS/Linux:-
python3 -m venv venv
source venv/bin/activate


🔹 Install required packages:-

pip install -r requirements.txt


Run Locally:-

streamlit run app.py

face-segmentation/
├── app.py                # Streamlit UI
├── face_segment_fin.py   # Processing logic (model, segmentation)
├── model.py              # BiSeNet architecture
├── requirements.txt      # Python dependencies
├── README.md             # This file



