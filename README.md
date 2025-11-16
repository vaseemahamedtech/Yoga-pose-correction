🧘 Yoga Pose AI - ML & DL Powered Pose Correction System
Complete yoga pose detection and correction system with Machine Learning and Deep Learning models, real-time analysis, and dataset integration.

📋 Table of Contents
Features

Project Structure

Installation

Dataset Setup

Training Models

Running the Application

Model Architecture

API Documentation

Model Performance

Customization

Troubleshooting

License

Acknowledgments

Support

✨ Features
🤖 Machine Learning Models
Random Forest - Ensemble learning with decision trees

Support Vector Machine (SVM) - Kernel-based classification

XGBoost - Gradient boosting for high performance

K-Nearest Neighbors - Instance-based learning

Decision Tree - Simple interpretable model

Naive Bayes - Probabilistic classifier

Logistic Regression - Linear classification baseline

🧠 Deep Learning Models
Dense Neural Network - Multi-layer perceptron with dropout

1D CNN - Convolutional network for sequence data

LSTM - Long Short-Term Memory for temporal patterns

Hybrid CNN-LSTM - Combined architecture for best performance

📊 Complete ML Pipeline
Dataset Retrieval - Download from Kaggle and other sources

Preprocessing - Image processing and landmark extraction

Feature Engineering - Angle, distance, and symmetry features

Model Selection - Multiple algorithms comparison

Hyperparameter Tuning - Optimization for best performance

Validation - Cross-validation and holdout testing

Evaluation - Comprehensive metrics and confusion matrices

🎯 Real-time Features
Live webcam pose analysis

Photo upload and analysis

Real-time feedback and corrections

Pose classification and scoring

Multiple pose support (Mountain, Warrior II, Tree, Downward Dog, etc.)

📁 Project Structure
Plaintext

YogaPoseML/
├── backend/
│   ├── main.py                   # FastAPI server
│   ├── data_preprocessor.py      # Dataset preprocessing
│   ├── feature_engineer.py       # Feature extraction
│   ├── ml_trainer.py             # ML model training
│   ├── dl_trainer.py             # DL model training
│   ├── pose_analyzer.py          # Pose analysis logic
│   ├── pose_classifier.py        # Pose classification
│   ├── enhanced_pose_analyzer.py # Advanced analysis
│   └── advanced_analyzer.py      # Expert system
├── frontend/
│   ├── src/
│   │   ├── App.js                # Main React component
│   │   ├── App.css
│   │   └── index.js
│   ├── components/
│   │   ├── LiveAnalysis.js       # Live camera component
│   │   ├── PhotoAnalysis.js      # Photo upload component
│   │   └── *.css
│   └── public/
│       └── index.html
├── dataset/
│   ├── raw/                      # Raw downloaded datasets
│   ├── processed/                # Preprocessed features
│   ├── train/                    # Training split
│   ├── test/                     # Test split
│   └── validation/               # Validation split
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks
├── utils/
│   └── dataset_downloader.py     # Dataset download helper
├── requirements.txt
└── README.md
🚀 Installation
1. Clone or Extract the Project
Bash

cd YogaPoseML
2. Create Virtual Environment (Recommended)
Bash

python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
📥 Dataset Setup
Option 1: Download Yoga-82 Dataset (Recommended)
Visit: https://sites.google.com/view/yoga-82/home

Fill the form to request download access

Extract to: dataset/raw/yoga82/

Option 2: Kaggle Datasets
Install Kaggle CLI:

Bash

pip install kaggle
Setup Kaggle credentials:

Go to https://www.kaggle.com/settings

Click "Create New API Token"

Save kaggle.json to ~/.kaggle/

Run: chmod 600 ~/.kaggle/kaggle.json

Download Yoga-16 Dataset:

Bash

kaggle datasets download -d mohiuddin2531/yoga-16
unzip yoga-16.zip -d dataset/raw/yoga16/
Download Yoga Poses Dataset (5 poses):

Bash

kaggle datasets download -d ujjwalchowdhury/yoga-pose-classification
unzip yoga-pose-classification.zip -d dataset/raw/yoga5/
Option 3: Get Download Instructions
Bash

python utils/dataset_downloader.py
🎓 Training Models
Step 1: Preprocess Dataset
Bash

python backend/data_preprocessor.py
This will:

Extract pose landmarks using MediaPipe

Create feature vectors (angles, distances, symmetry)

Split into train/validation/test sets (70/20/10)

Normalize features

Save processed data

Step 2: Train ML Models
Bash

python backend/ml_trainer.py
Trains:

Random Forest

SVM

XGBoost

K-NN

Decision Tree

Naive Bayes

Logistic Regression

Results saved to models/ directory.

Step 3: Train DL Models
Bash

python backend/dl_trainer.py
Trains:

Dense Neural Network

1D CNN

LSTM

Hybrid CNN-LSTM

Best models saved with checkpoints.

🖥️ Running the Application
Start Backend Server
Bash

cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
API will be available at: http://localhost:8000

Start Frontend (Optional)
Bash

cd frontend
npm install
npm start
Frontend will be available at: http://localhost:3000

Test the API
Health Check:

Bash

curl http://localhost:8000/health
Analyze Photo:

Bash

curl -X POST -F "file=@yoga_image.jpg" -F "pose_type=mountain" \
  http://localhost:8000/analyze-photo/
Get Supported Poses:

Bash

curl http://localhost:8000/supported-poses
🏗️ Model Architecture
ML Pipeline
Plaintext

Raw Image
    ↓
MediaPipe Pose Detection
    ↓
Landmark Extraction (33 points × 4 coords = 132 features)
    ↓
Feature Engineering
    ├── Angle Features (6 features)
    ├── Distance Features (3 features)
    └── Symmetry Features (3 features)
    ↓
Feature Vector (144 features)
    ↓
StandardScaler Normalization
    ↓
ML Classifier
    ↓
Pose Prediction + Confidence
DL Architecture
Dense Neural Network:

Plaintext

Input (132) → Dense(256) → BatchNorm → Dropout(0.4) →
Dense(128) → BatchNorm → Dropout(0.3) →
Dense(64) → BatchNorm → Dropout(0.2) →
Dense(32) → Dropout(0.1) →
Output (num_classes) Softmax
CNN Architecture:

Plaintext

Input (132, 1) → Conv1D(64) → BatchNorm → MaxPool → Dropout →
Conv1D(128) → BatchNorm → MaxPool → Dropout →
Conv1D(256) → BatchNorm → GlobalAvgPool →
Dense(128) → Dropout → Output Softmax
LSTM Architecture:

Plaintext

Input (132) → Reshape(33, 4) →
LSTM(128, return_seq) → Dropout →
LSTM(64, return_seq) → Dropout →
LSTM(32) → Dropout →
Dense(64) → Dropout → Output Softmax
📡 API Documentation
Endpoints
GET /
Root endpoint with API information

GET /health
Health check endpoint

GET /supported-poses
Get list of supported yoga poses

POST /analyze-photo/
Analyze yoga pose from uploaded image

Parameters:

file: Image file (multipart/form-data)

pose_type: Expected pose type (optional)

Response:

JSON

{
  "status": "success",
  "analysis": {
    "score": 85,
    "feedback": ["Feedback 1", "Feedback 2"],
    "pose_type": "mountain",
    "pose_name": "Mountain Pose",
    "detailed_metrics": {
      "alignment_score": 80,
      "balance_score": 75,
      "form_score": 85,
      "stability_score": 82
    }
  },
  "annotated_image": "base64_string",
  "landmarks_detected": true,
  "timestamp": "2024-10-22T12:00:00"
}
WebSocket /ws/live-analysis
Real-time webcam analysis

Send: Base64 encoded image frames

Receive: Real-time analysis results

📊 Model Performance
After training, models are evaluated on test set with metrics:

Accuracy - Overall classification accuracy

Precision - True positives / (True positives + False positives)

Recall - True positives / (True positives + False negatives)

F1-Score - Harmonic mean of precision and recall

Confusion Matrix - Detailed class-wise performance

Results are saved to models/ directory.

🛠️ Customization
Add New Pose
Add images to dataset/raw/your_pose/

Run preprocessing: python backend/data_preprocessor.py

Retrain models: python backend/ml_trainer.py

Update main.py to include new pose

Adjust Feature Engineering
Edit backend/feature_engineer.py to:

Add new angle calculations

Include additional distance metrics

Modify symmetry features

Hyperparameter Tuning
Edit backend/ml_trainer.py or backend/dl_trainer.py:

Adjust model parameters

Change learning rates

Modify batch sizes

Update epochs

🐛 Troubleshooting
Issue: Dataset not found
Solution: Run python utils/dataset_downloader.py for instructions

Issue: MediaPipe not working
Solution: Install system dependencies:

Bash

# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# macOS
brew install opencv
Issue: TensorFlow GPU not detected
Solution: Install CUDA and cuDNN compatible versions

Issue: Port already in use
Solution: Change port in command:

Bash

uvicorn main:app --reload --port 8001
📝 License
This project is for educational purposes. Datasets may have their own licenses.

🙏 Acknowledgments
MediaPipe by Google

Yoga-82 Dataset by IIT Gandhinagar

Kaggle Community for datasets

OpenCV and TensorFlow teams

📧 Support
For issues, feature requests, or questions, please open an issue in the repository.

Happy Coding! 🧘‍♀️🤖#   Y o g a - p o s e - c o r r e c t i o n  
 