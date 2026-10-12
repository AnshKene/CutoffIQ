import os
import sys
import pickle
import numpy as np
import pandas as pd

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
    sklearn.impute.SimpleImputer._fill_dtype = property(lambda self: getattr(self, '_fit_dtype', None))

# BitGenerator unpickling shim for NumPy compatibility across versions
try:
    import numpy.random._pickle
    _orig_bit_gen_ctor = numpy.random._pickle.__bit_generator_ctor
    def _patched_bit_gen_ctor(bit_gen='MT19937'):
        if isinstance(bit_gen, type):
            return bit_gen()
        name = getattr(bit_gen, '__name__', str(bit_gen))
        if name in numpy.random._pickle.BitGenerators:
            return numpy.random._pickle.BitGenerators[name]()
        return _orig_bit_gen_ctor(bit_gen)
    numpy.random._pickle.__bit_generator_ctor = _patched_bit_gen_ctor
except Exception:
    pass


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


# Register classes in module contexts for pickle deserialization
sys.modules['__main__'].TargetEncoderRegressor = TargetEncoderRegressor
sys.modules['__main__'].CutoffEnsembleRegressor = CutoffEnsembleRegressor
if 'app' not in sys.modules:
    sys.modules['app'] = sys.modules[__name__]


# ---------------------------------------------------------
# Resource Loading
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')


def load_model_and_scaler():
    loaded_model = None
    loaded_scaler = None
    model_err = None

    if os.path.exists(MODEL_PATH):
        # Auto-detect unpulled Git LFS pointer file (< 2 KB text file)
        try:
            file_size = os.path.getsize(MODEL_PATH)
            if file_size < 2048:
                with open(MODEL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read(512)
                if 'git-lfs' in txt:
                    print("Detected Git LFS pointer for model.pkl. Pulling via git lfs pull...")
                    import subprocess
                    subprocess.run(["git", "lfs", "pull"], check=False)
        except Exception as lfs_e:
            print(f"Git LFS check notice: {lfs_e}")

        try:
            with open(MODEL_PATH, 'rb') as f:
                loaded_model = pickle.load(f)
            print("CutoffIQ Ensemble Model loaded successfully!")
        except Exception as e:
            model_err = str(e)
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        model_err = f"File not found at {MODEL_PATH}"

    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                loaded_scaler = pickle.load(f)
            print("StandardScaler loaded successfully!")
        except Exception as e:
            print(f"Note: Scaler load warning: {e}")

    return loaded_model, loaded_scaler, model_err


# ---------------------------------------------------------
# Flask API Support (for Docker / Gunicorn backend deployment)
# ---------------------------------------------------------
try:
    from flask import Flask, request as flask_request, jsonify
    flask_app = Flask(__name__)
    app = flask_app  # WSGI entrypoint for Gunicorn (app:app)

    @flask_app.route('/', methods=['GET'])
    def health():
        m, s, _ = load_model_and_scaler()
        return jsonify({
            'status': 'healthy',
            'service': 'CutoffIQ API',
            'version': '1.0.0',
            'model_loaded': m is not None,
            'scaler_loaded': s is not None,

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

    @flask_app.route('/predict', methods=['POST'])
    def predict_endpoint():
        m, _, _ = load_model_and_scaler()
        if m is None:
            return jsonify({'status': 'error', 'message': 'Model is not loaded.'}), 500

        try:
            data = flask_request.get_json(force=True)
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                return jsonify({'status': 'error', 'message': 'Request must be JSON object or array.'}), 400

            df_input = pd.DataFrame(data)
            if 'year' in df_input.columns:
                df_input['year'] = pd.to_numeric(df_input['year'], errors='coerce').fillna(2024)
            if 'round' in df_input.columns:
                df_input['round'] = df_input['round'].astype(str)

            preds = m.predict(df_input)
            return jsonify({
                'status': 'success',
                'count': len(preds),
                'predictions': [round(float(p), 2) for p in preds]
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

except ImportError:
    flask_app = None
    app = None


# ---------------------------------------------------------
# Streamlit Frontend Web Application
# ---------------------------------------------------------
def render_streamlit_frontend():
    import streamlit as st

    st.set_page_config(
        page_title="CutoffIQ | AI Admission Cutoff Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .prediction-card {
            background: linear-gradient(135deg, rgba(37,99,235,0.08) 0%, rgba(124,58,237,0.12) 100%);
            border: 1.5px solid rgba(124,58,237,0.25);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-top: 15px;
            margin-bottom: 20px;
        }
        .prediction-score {
            font-size: 3.2rem;
            font-weight: 800;
            color: #1E40AF;
            letter-spacing: -1px;
            margin: 8px 0;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 9999px;
            background-color: #EEF2FF;
            color: #4F46E5;
            margin-bottom: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Loading CutoffIQ Ensemble Model...")
    def get_model_and_scaler():
        return load_model_and_scaler()

    model, scaler, model_err = get_model_and_scaler()

    # Extract vocabulary from target encoder if available
    known_unis = []
    known_programmes = []
    known_colleges = []
    known_variants = []
    known_categories = []
    known_seat_types = []

    if model is not None and hasattr(model, 'target_encoder'):
        maps = getattr(model.target_encoder, 'maps', {})
        known_unis = sorted(list(maps.get('university', {}).keys()))
        known_programmes = sorted(list(maps.get('programme', {}).keys()))
        known_colleges = sorted(list(maps.get('college', {}).keys()))
        known_variants = sorted(list(maps.get('variant', {}).keys()))
        known_categories = sorted(list(maps.get('category', {}).keys()))
        known_seat_types = sorted(list(maps.get('seat_type', {}).keys()))

    # Fallback lists if empty
    if not known_unis:
        known_unis = ["Banaras Hindu University", "Delhi University", "Central University of Jammu"]
    if not known_categories:
        known_categories = ["UR", "OBC", "SC", "ST", "EWS", "CW", "PWD"]
    if not known_seat_types:
        known_seat_types = ["All-India open merit", "Paid Fee Course", "general"]

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Architecture")
        st.markdown("""
        **CutoffIQ Blended Ensemble**:
        - 🌲 **HistGradientBoosting** (74% weight): Fast tree boosting on smoothed target-encoded categories.
        - 🌳 **RandomForest Pipeline** (26% weight): Imputation, standard scaling, and variance mitigation.
        """)
        
        st.divider()
        st.markdown("### ⚡ Quick Presets")
        preset = st.radio(
            "Select a prefilled sample:",
            ["Default", "BHU Pharmacy", "Delhi Commerce", "Custom Input"],
            index=0
        )
        
        st.divider()
        st.markdown("### 📡 API Integration")
        with st.expander("Show REST API curl"):
            st.code("""
curl -X POST https://your-domain/predict \\
  -H "Content-Type: application/json" \\
  -d '[{
    "university": "Banaras Hindu University",
    "year": 2024,
    "programme": "2-year Diploma in Pharmacy",
    "college": "Acharya Narendra Dev College",
    "variant": "Accounting and Finance + Economics",
    "round": "1",
    "category": "CW",
    "seat_type": "Paid Fee Course"
  }]'
            """, language="bash")

    # Main Header
    st.markdown('<span class="badge">AI-Powered Admission Intelligence</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">CutoffIQ 🎓 Cutoff Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Estimate university and college admission cutoff scores across diverse quotas, programmes, and rounds.</div>', unsafe_allow_html=True)

    if model is None:
        file_sz = f"{os.path.getsize(MODEL_PATH):,} bytes" if os.path.exists(MODEL_PATH) else "Not found"
        st.error(f"⚠️ **Model file (`model.pkl`) could not be loaded**: `{model_err}`\n\n*(Detected file size on disk: {file_sz})*")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔄 Pull Git LFS & Reload Model", type="primary", use_container_width=True):
                import subprocess
                subprocess.run(["git", "lfs", "pull"], check=False)
                st.cache_resource.clear()
                st.rerun()
        with col_b2:
            if st.button("🧹 Clear Streamlit Cache & Retry", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        return

    # Tabs
    tab_single, tab_batch, tab_info = st.tabs(["🎯 Single Prediction", "📁 Batch CSV Predictor", "ℹ️ About the Model"])

    # Preset values handler
    default_uni = known_unis[0]
    default_prog = known_programmes[0] if known_programmes else "2-year Diploma in Pharmacy"
    default_college = known_colleges[0] if known_colleges else "Acharya Narendra Dev College"
    default_variant = known_variants[0] if known_variants else "Accounting and Finance + Economics"
    default_round = "1"
    default_cat = "UR" if "UR" in known_categories else known_categories[0]
    default_seat = known_seat_types[0]

    if preset == "BHU Pharmacy":
        default_uni = "Banaras Hindu University"
        default_prog = "2-year Diploma in Pharmacy" if "2-year Diploma in Pharmacy" in known_programmes else default_prog
        default_cat = "CW" if "CW" in known_categories else default_cat
        default_seat = "Paid Fee Course" if "Paid Fee Course" in known_seat_types else default_seat
    elif preset == "Delhi Commerce":
        if any("Delhi" in u for u in known_unis):
            default_uni = next(u for u in known_unis if "Delhi" in u)
        default_cat = "OBC" if "OBC" in known_categories else default_cat

    with tab_single:
        with st.form("cutoff_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏛️ Institution & Course")
                uni_idx = known_unis.index(default_uni) if default_uni in known_unis else 0
                university = st.selectbox("University", known_unis, index=uni_idx)

                prog_idx = known_programmes.index(default_prog) if default_prog in known_programmes else 0
                programme = st.selectbox("Academic Programme", known_programmes, index=prog_idx)

                college_idx = known_colleges.index(default_college) if default_college in known_colleges else 0
                college = st.selectbox("Affiliated College", known_colleges, index=college_idx)

                variant_idx = known_variants.index(default_variant) if default_variant in known_variants else 0
                variant = st.selectbox("Subject Combination / Variant", known_variants, index=variant_idx)

            with col2:
                st.markdown("#### 🎫 Admission Parameters")
                year = st.number_input("Academic Year", min_value=2020, max_value=2030, value=2024, step=1)
                
                round_options = [str(r) for r in range(1, 16)]
                round_val = st.selectbox("Counselling Round", round_options, index=0)

                cat_idx = known_categories.index(default_cat) if default_cat in known_categories else 0
                category = st.selectbox("Quota Category", known_categories, index=cat_idx)

                seat_idx = known_seat_types.index(default_seat) if default_seat in known_seat_types else 0
                seat_type = st.selectbox("Seat Allocation Type", known_seat_types, index=seat_idx)

            submitted = st.form_submit_button("🔮 Predict Cutoff Score", type="primary", use_container_width=True)

        if submitted:
            input_data = [{
                'university': university,
                'year': year,
                'programme': programme,
                'college': college,
                'variant': variant,
                'round': str(round_val),
                'category': category,
                'seat_type': seat_type
            }]

            df_single = pd.DataFrame(input_data)
            try:
                with st.spinner("Computing blended ensemble prediction..."):
                    pred = model.predict(df_single)[0]
                    pred_score = round(float(pred), 2)

                st.markdown(f"""
                <div class="prediction-card">
                    <span class="badge">Predicted Cutoff Score</span>
                    <div class="prediction-score">{pred_score}</div>
                    <p style="color: #64748B; margin: 0;">Target score estimated for <b>{programme}</b> at <b>{university}</b> ({category} - Round {round_val})</p>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Category", category)
                col_b.metric("Round", f"Round {round_val}")
                col_c.metric("Ensemble Calibration", "74% HGB + 26% RF")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_batch:
        st.markdown("#### 📁 Batch Processing via CSV")
        st.write("Upload a CSV file containing columns: `university`, `year`, `programme`, `college`, `variant`, `round`, `category`, `seat_type`.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(df_upload)}** rows. Preview:")
                st.dataframe(df_upload.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        preds = model.predict(df_upload)
                        df_upload['predicted_cutoff'] = [round(float(p), 2) for p in preds]

                    st.success(f"Successfully generated predictions for {len(df_upload)} applicants!")
                    st.dataframe(df_upload[['university', 'programme', 'category', 'round', 'predicted_cutoff']].head(10), use_container_width=True)

                    csv_result = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Predictions as CSV",
                        data=csv_result,
                        file_name="cutoffiq_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

    with tab_info:
        st.markdown("""
        ### 🧠 How CutoffIQ Works
        1. **High-Cardinality Target Encoding**: Categorical features with hundreds of colleges and combinations are smoothed to avoid target leakage while preserving high predictive signals.
        2. **HistGradientBoosting**: Fast, bin-based gradient boosted decision trees capturing non-linear interactions across features.
        3. **RandomForest Pipeline**: Ensembles parallelized tree estimators with imputation to prevent over-fitting.
        4. **Blended Ensemble**: Combines both predictors (74% HGB + 26% RF) to output stable, reliable cutoff scores.
        """)


# Execution Router: Streamlit or Flask
try:
    import streamlit as st
    if st.runtime.exists():
        render_streamlit_frontend()
except Exception:
    pass

if __name__ == '__main__':
    # When executed directly with `python app.py` outside Streamlit runtime
    try:
        import streamlit as st
        is_streamlit = st.runtime.exists()
    except Exception:
        is_streamlit = False

    if not is_streamlit and flask_app is not None:
        print("Starting Flask API server on port 5000...")
        print("💡 To launch the interactive Web UI instead, run: streamlit run app.py")
        flask_app.run(host='0.0.0.0', port=5000, debug=False)