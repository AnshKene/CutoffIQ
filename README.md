# CutoffIQ 🎓📊

**CutoffIQ** is a machine learning inference microservice designed to predict university and college admission cutoff scores based on multi-dimensional admission parameters (such as institution, academic programme, quota category, round, and seat allocation type).

---

## 🚀 Key Features

- **Blended Ensemble Modeling**: Combines gradient boosting and ensemble trees with weighted calibration.
  - **HistGradientBoostingRegressor** (~74% weight): High-speed tree-based regressor operating on smoothed target-encoded categorical features.
  - **RandomForestRegressor Pipeline** (~26% weight): Pipeline combining standard scaling, imputation, and one-hot encoding for robust generalization against variance.
- **RESTful Flask API**: Lightweight, production-ready inference endpoints supporting single and batch JSON inputs.
- **Backward & Forward Compatibility**: Includes custom deserialization shims allowing seamless model unpickling across different scikit-learn runtime versions.
- **Containerized Deployment**: Includes ready-to-run `Dockerfile` powered by Gunicorn WSGI.

---

## 🏛️ Model Architecture

```text
                  +-------------------------+
                  |    Input Features JSON  |
                  +-------------------------+
                               |
               +---------------+---------------+
               |                               |
               v                               v
    +--------------------+           +--------------------+
    |   TargetEncoder    |           | ColumnTransformer  |
    | (Smoothing = 20)   |           | (OHE + Scaler)     |
    +--------------------+           +--------------------+
               |                               |
               v                               v
    +--------------------+           +--------------------+
    | HistGradientBoost  |           |   RandomForest     |
    |     Regressor      |           |     Pipeline       |
    +--------------------+           +--------------------+
               |                               |
         (Weight: 0.74)                  (Weight: 0.26)
               \                               /
                \                             /
                 v                           v
              +---------------------------------+
              |    Weighted Blended Prediction  |
              +---------------------------------+
```

### Input Features

| Feature | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `university` | string | Name of the university | `"Banaras Hindu University"` |
| `year` | integer | Admission academic year | `2024` |
| `programme` | string | Name of the degree / course | `"2-year Diploma in Pharmacy"` |
| `college` | string | Affiliated college name | `"Acharya Narendra Dev College"` |
| `variant` | string | Subject combination / variant | `"Accounting and Finance + Economics"` |
| `round` | string/int | Counselling admission round | `"1"` |
| `category` | string | Reservation category | `"CW"`, `"OBC"`, `"UR"` |
| `seat_type` | string | Seat quota type | `"Paid Fee Course"`, `"All-India open merit"` |

---

## 📦 Project Structure

```text
CutoffIQ/
├── app.py             # Flask application & ensemble prediction logic
├── model.pkl          # Serialized ensemble model weights (Git LFS)
├── scaler.pkl         # Serialized StandardScaler
├── requirements.txt   # Python dependency specifications
├── Dockerfile         # Docker container configuration
├── test_api.py        # Verification & test client script
├── .gitattributes     # Git LFS configuration for large model files
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```

---

## ⚡ Quick Start

### 1. Prerequisites & Git LFS
Since `model.pkl` is > 100 MB, ensure **Git LFS** is installed before cloning:
```bash
git lfs install
git clone https://github.com/AnshKene/CutoffIQ.git
cd CutoffIQ
git lfs pull
```

### 2. Environment Setup
Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Development Server
```bash
python app.py
```
The server will start at `http://localhost:5000`.

---

## 📡 API Reference

### Health Check
- **Endpoint**: `GET /`
- **Response**:
```json
{
  "service": "CutoffIQ API",
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### Predict Cutoffs
- **Endpoint**: `POST /predict`
- **Headers**: `Content-Type: application/json`
- **Request Body** (Single or Array):
```json
[
  {
    "university": "Banaras Hindu University",
    "year": 2024,
    "programme": "2-year Diploma in Pharmacy",
    "college": "Acharya Narendra Dev College",
    "variant": "Accounting and Finance + Economics",
    "round": "1",
    "category": "CW",
    "seat_type": "Paid Fee Course"
  }
]
```
- **Response**:
```json
{
  "status": "success",
  "count": 1,
  "predictions": [274.08]
}
```

---

## 🐳 Docker Deployment

Build and run using Docker:
```bash
docker build -t cutoffiq:latest .
docker run -p 5000:5000 cutoffiq:latest
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
