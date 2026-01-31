# 🐮 Cow Breed Classification App

A Streamlit-based web application that identifies cow breeds using a deep learning model (EfficientNet-B0).  
It also provides market information and breed insights based on prediction.

---

## 🚀 Features
- Upload cow images (JPG/PNG)
- Classifies breed using EfficientNet-B0
- Displays breed metadata from JSON file
- Simple and fast Streamlit UI
- Runs fully on CPU

---

## 🧠 Model
- Architecture: EfficientNet-B0  
- Framework: PyTorch  
- Input size: 224 × 224  
- Trained on a custom dataset  
- Outputs breed labels using `labels.txt`

Model file used: `best_model.pth`

---

## 🛠️ Technologies Used
- Streamlit
- PyTorch
- Torchvision
- Pillow
- JSON

---

## 📦 Installation (if running locally)

```bash
pip install -r requirements.txt
streamlit run app.py

## 📁 Project Structure

```
my-streamlit-app/
│── app.py
│── best_model.pth
│── labels.txt
│── breed_info.json
│── requirements.txt
└── README.md
```



