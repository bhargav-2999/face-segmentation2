# 🧠 Face Segmentation App

A web app that performs **face and hair segmentation** on images using deep learning. Upload an image containing exactly **one face**, and receive a transparent background version with only the segmented head/face area.

---

## 🚀 Features

- ✅ Validates that image has only **one human face**
- 🎯 Uses **BiSeNet** for high-quality face segmentation
- 🎨 Outputs RGBA image with a **transparent background*

---

## 💻 Setup Instructions

### 🔹Download the zip folder and unzip it 

### 🔹 then open the unzipped folder in vs code then cd face-segmentation2-main

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
├── app.py                 # Streamlit UI
├── face_segment_fin.py    # Core logic (model + preprocessing + auto model download)
├── model.py               # BiSeNet architecture
├── 79999_iter.pth         # 🔥 Model weights
├── requirements.txt       # All required libraries
├── README.md              # Full documentation
└── .gitignore             # Prevent .pth/.png files from being tracked




