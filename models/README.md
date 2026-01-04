# Models Directory

This directory stores trained ML and DL models.

## ML Models

After running `python backend/ml_trainer.py`, you'll find:

- `best_ml_Random_Forest_model.pkl` - Random Forest classifier
- `best_ml_XGBoost_model.pkl` - XGBoost classifier
- `best_ml_SVM_model.pkl` - Support Vector Machine
- `label_encoder.pkl` - Label encoder for class names

## DL Models

After running `python backend/dl_trainer.py`, you'll find:

- `dense_nn_best.h5` - Dense Neural Network
- `cnn_1d_best.h5` - 1D CNN
- `lstm_best.h5` - LSTM model
- `hybrid_cnn_lstm_best.h5` - Hybrid CNN-LSTM
- `dl_results.json` - Training results

## Usage

Models are automatically loaded by the main application.

To use a specific model:

```python
import pickle
with open('models/best_ml_Random_Forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

For DL models:

```python
from tensorflow import keras
model = keras.models.load_model('models/dense_nn_best.h5')
```
