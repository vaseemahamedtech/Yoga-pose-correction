Got it! I’ll **trim your Yoga Pose AI README** to include **only the essential, necessary content** while keeping it professional, clear, and concise. Here’s the cleaned-up version:

---

# Yoga Pose AI: ML & DL Powered Pose Correction System

A real-time yoga pose detection and correction system powered by Machine Learning and Deep Learning models. Supports multiple yoga poses, provides live webcam analysis, and gives actionable feedback for pose improvement.

---

## ✨ Features

* **Machine Learning Models:** Random Forest, SVM, XGBoost, K-NN, Decision Tree, Logistic Regression
* **Deep Learning Models:** Dense Neural Network, CNN, LSTM, Hybrid CNN-LSTM
* **Real-time Pose Analysis:** Webcam or uploaded image input
* **Pose Feedback:** Alignment, balance, stability, and form metrics
* **Supports Multiple Poses:** Mountain, Warrior II, Tree, Downward Dog, etc.

---

## 📁 Project Structure

```
YogaPoseML/
├── backend/                # FastAPI backend
│   ├── main.py
│   ├── data_preprocessor.py
│   ├── ml_trainer.py
│   └── dl_trainer.py
├── frontend/               # React frontend
│   ├── src/
│   └── public/
├── dataset/                # Raw and processed datasets
├── models/                 # Trained ML/DL models
├── utils/                  # Helpers (e.g., dataset downloader)
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd YogaPoseML
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

1. Download Yoga-82 or Kaggle datasets and extract to:
   `dataset/raw/yoga82/` or `dataset/raw/yoga16/`
2. Optional: Use downloader script:

```bash
python utils/dataset_downloader.py
```

---

## 🎓 Training Models

* **Preprocess dataset:**

```bash
python backend/data_preprocessor.py
```

* **Train ML models:**

```bash
python backend/ml_trainer.py
```

* **Train DL models:**

```bash
python backend/dl_trainer.py
```

---

## 🖥️ Running the Application

* **Start backend:**

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

* **Start frontend (optional):**

```bash
cd frontend
npm install
npm start
```

* Backend API available at: `http://localhost:8000`
* Frontend available at: `http://localhost:3000`

---

## 📡 API Endpoints

* `GET /health` → Health check
* `GET /supported-poses` → List of supported poses
* `POST /analyze-photo/` → Analyze uploaded image

  * **Parameters:** `file` (image), `pose_type` (optional)
  * **Response:** JSON with scores, feedback, and landmarks

---

## 🛠️ Customization

* Add new poses by adding images, preprocessing, and retraining models
* Adjust feature extraction in `backend/feature_engineer.py`
* Tune hyperparameters in `ml_trainer.py` or `dl_trainer.py`

---

## 📝 License

This project is for **educational purposes only**. Datasets may have their own licenses.

---

## 🙏 Acknowledgments

* MediaPipe by Google
* Yoga-82 Dataset by IIT Gandhinagar
* Kaggle Community
* OpenCV & TensorFlow teams

---

**Happy Coding! 🧘‍♂️🤖**
