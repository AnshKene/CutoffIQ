import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ---------------------------------------------------------
# Compatibility Shims for Scikit-Learn Pipeline Deserialization
# ---------------------------------------------------------
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        """Compatibility shim for scikit-learn models serialized with earlier versions."""
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import sklearn.impute
if not hasattr(sklearn.impute.SimpleImputer, '_fill_dtype'):
    # In scikit-learn >= 1.7, SimpleImputer uses _fill_dtype, whereas older versions used _fit_dtype
    sklearn.impute.SimpleImputer._fill_dtype = property(lambda self: getattr(self, '_fit_dtype', None))


# ---------------------------------------------------------
# Custom Model & Preprocessing Classes
# ---------------------------------------------------------
class TargetEncoderRegressor:
    """Target encoder used during preprocessing for high-cardinality categorical features."""
    def __init__(self, cols=None, smoothing=20):
        self.cols = cols or []
        self.smoothing = smoothing
        self.maps = {}
        self.global_mean = 0.0

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            if col in X_out.columns:
                col_map = self.maps.get(col, {})
                X_out[col] = X_out[col].map(col_map).fillna(self.global_mean)
        return X_out


class CutoffEnsembleRegressor:
    """
    Blended Ensemble Regressor combining HistGradientBoostingRegressor
    and RandomForestRegressor Pipeline.
    """
    def __init__(self, target_encoder=None, hgb_model=None, rf_model=None, hgb_weight=0.74):
        self.target_encoder = target_encoder
        self.hgb_model = hgb_model
        self.rf_model = rf_model
        self.hgb_weight = hgb_weight

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Feature preparation & Target Encoding for HistGradientBoosting
        X_encoded = self.target_encoder.transform(X)
        hgb_features = ['university', 'year', 'programme', 'college', 'variant', 'round', 'category', 'seat_type']
        
        # Ensure all required features are present
        for feat in hgb_features:
            if feat not in X_encoded.columns:
                X_encoded[feat] = self.target_encoder.global_mean if feat != 'year' else 2024

        X_hgb = X_encoded[hgb_features].values.astype(float)
        hgb_pred = self.hgb_model.predict(X_hgb)

        # 2. Predict with RandomForest Pipeline
        rf_pred = self.rf_model.predict(X)

        # 3. Weighted Ensemble Blend (0.74 HGB + 0.26 RF)
        blended_preds = self.hgb_weight * hgb_pred + (1.0 - self.hgb_weight) * rf_pred
        return blended_preds


# Ensure classes are registered in __main__ context for pickle loading
sys.modules['__main__'].TargetEncoderRegressor = TargetEncoderRegressor
sys.modules['__main__'].CutoffEnsembleRegressor = CutoffEnsembleRegressor


# ---------------------------------------------------------
# Flask Application Setup
# ---------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = None
scaler = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("CutoffIQ Ensemble Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")

try:
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("StandardScaler loaded successfully!")
except Exception as e:
    print(f"Note: Scaler load warning: {e}")


@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'CutoffIQ API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'endpoints': {
            '/predict': {
                'method': 'POST',
                'description': 'Predict university admission cutoff scores',
                'expected_fields': [
                    'university', 'year', 'programme', 'college',
                    'variant', 'round', 'category', 'seat_type'
                ]
            }
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model is not loaded. Please verify model.pkl.'
        }), 500

    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            return jsonify({
                'status': 'error',
                'message': 'Request body must be a JSON object or list of objects.'
            }), 400

        df_input = pd.DataFrame(data)

        # Basic type casting and safety checks
        if 'year' in df_input.columns:
            df_input['year'] = pd.to_numeric(df_input['year'], errors='coerce').fillna(2024)
        if 'round' in df_input.columns:
            df_input['round'] = df_input['round'].astype(str)

        # Generate predictions using the blended ensemble pipeline
        predictions = model.predict(df_input)

        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'predictions': [round(float(p), 2) for p in predictions]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)