import tensorflow
import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

st.set_page_config("Exoplanet Hunter", layout='wide')

with st.sidebar:
    st.title("🪐 About")
    st.markdown("""
    **Exoplanet Hunt** uses a 1D Convolutional Neural Network trained on NASA's 
    Kepler space telescope data to detect exoplanet candidates from stellar 
    light curves.
    
    ---
    **Model:** Baseline 1D CNN  
    **Dataset:** Kepler Labelled Time Series  
    **Input:** 3197 flux measurements per star  
    **Metric:** PR-AUC (imbalanced classification)
    
    ---
    **How it works:**  
    A planet passing in front of its star causes a tiny dip in brightness. 
    The model learns to recognize these transit patterns from raw flux signals.
    
    ---
    **⚠️ Limitation:**  
    Only ~37 positive training examples exist — overfitting is a known 
    constraint of this dataset.

    ---         
    **📋 Compatible formats:
    - Kepler labelled time series (Kaggle) — exactly 3197 flux columns
    - Longer light curves — handled via sliding window
    - Shorter than 3197 points — not supported (model requires minimum 3197 flux values)
    
    ---
    🔗 [View on GitHub](https://github.com/ANSHAB786/exoplanet-detection-kepler)
    """)
 
    st.markdown("---")
 
    # Sample CSV download
    st.subheader("🧪 Try it out")
    st.caption("No data? Download a sample light curve to test the app.")
 
    # Generate a synthetic sample CSV (one planet-like star + one normal star)
    np.random.seed(42)
    n_flux = 3197
    # Normal star — flat with noise
    normal = np.random.normal(0, 0.01, n_flux)
    # Planet-like star — flat with a dip at position 1000
    planet = np.random.normal(0, 0.01, n_flux)
    planet[1000:1020] -= 0.15  # transit dip
 
    sample_df = pd.DataFrame(
        [normal, planet],
        columns=[f"FLUX.{i+1}" for i in range(n_flux)]
    )
    sample_csv = sample_df.to_csv(index=False)
    st.download_button(
        "📥 Download Sample CSV",
        sample_csv,
        "sample_lightcurves.csv",
        help="2 stars: 1 normal, 1 with a simulated transit dip"
    )
 
    st.markdown("---")
    if st.button("🗑️ Clear cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
#-Main
st.title("🔭 Exoplanet Hunt")
st.write("🪐 Got a light curve? Let's check for planets.")
st.caption("Upload your raw light curve CSV — the model will scan each star for transit signals.")
 
@st.cache_resource
def load_exoplanet_model():
    with st.spinner("🔁 Loading model..."):
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Exoplanet.best.keras')
        return load_model(MODEL_PATH)
 
model_cnn = load_exoplanet_model()
st.success("✔️ Model ready!")
 
uploaded = st.file_uploader("Choose a CSV file", type=['csv'])
 
if uploaded:
    # ── File decoding ──────────────────────────────────────────────────────────
    file_decode = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1', 'utf-16']
    df = None
    used_encoding = None
    for encoding in file_decode:
        try:
            df = pd.read_csv(uploaded, encoding=encoding)
            used_encoding = encoding
            break
        except Exception:
            continue
 
    if df is None:
        st.error("Could not read file. Please check the format.")
        st.stop()
 
    st.caption(f"File loaded with encoding: `{used_encoding}`")
 
    # Drop non-numeric columns (LABEL, ID, etc.)
    numeric_df = df.select_dtypes(include=[np.number])
 
    if numeric_df.empty:
        st.error("No numeric data found in file.")
        st.stop()
 
    # ── Multi-star mode ────────────────────────────────────────────────────────
    if len(numeric_df) > 1:
        st.info(f"📡 Processing {len(numeric_df)} stars...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
 
        for i in range(len(numeric_df)):
            status_text.text(f"Analyzing star {i+1} of {len(numeric_df)}...")
            progress_bar.progress((i + 1) / len(numeric_df))
 
            data = numeric_df.iloc[i].values[:3197]
            data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)
            x_input = data_norm.reshape(1, -1, 1)
            prob = model_cnn.predict(x_input, verbose=0)[0][0]
 
            results.append({
                'Star': i + 1,
                'Probability': f"{prob:.1%}",
                'Confidence': prob,
                'Result': '🪐 PLANET' if prob > 0.6 else '🌠 No Planet'
            })
 
        status_text.text("✔️ Analysis Complete")
        progress_bar.progress(1.0)
 
        result_df = pd.DataFrame(results)
        planet_df = result_df[result_df['Result'] == '🪐 PLANET']
        total_planets = len(planet_df)
 
        st.markdown("---")
        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stars", len(results))
        with col2:
            st.metric("🪐 Planets Found", total_planets)
        with col3:
            st.metric("🌠 Non-Planets", len(results) - total_planets)
 
        # Light curve preview
        st.markdown("---")
        st.subheader("📈 Light Curve Preview")
        preview_star = st.selectbox(
            "Select a star to preview its light curve",
            options=result_df['Star'].tolist(),
            format_func=lambda x: f"Star {x} — {result_df[result_df['Star']==x]['Result'].values[0]}"
        )
        selected_data = numeric_df.iloc[preview_star - 1].values[:3197]
        st.line_chart(selected_data)
 
        if total_planets > 0:
            st.markdown("---")
            st.subheader(f"🪐 Planet Candidates ({total_planets} found)")
            st.dataframe(planet_df[['Star', 'Probability', 'Result']])
 
        # Download
        csv_out = result_df[['Star', 'Probability', 'Result']].to_csv(index=False)
        st.download_button("📥 Download Results", csv_out, "results.csv", key="download_results")
 
    # ── Single-star mode ───────────────────────────────────────────────────────
    else:
        data = numeric_df.iloc[0].values
 
        if len(data) < 3197:
            st.error(f"❌ Need at least 3197 flux values. Your file has {len(data)} values.")
            st.stop()
 
        elif len(data) > 3197:
            st.info(f"📊 Longer light curve detected: {len(data)} values. Using sliding window...")
            window_size = 3197
            step_size = 300
            probabilities = []
            positions = []
 
            for start in range(0, len(data) - window_size + 1, step_size):
                window = data[start: start + window_size]
                window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
                x_input = window_norm.reshape(1, -1, 1)
                prob = model_cnn.predict(x_input, verbose=0)[0][0]
                probabilities.append(prob)
                positions.append(start)
 
            # Results after all windows
            best_idx = np.argmax(probabilities)
            best_prob = probabilities[best_idx]
            best_pos = positions[best_idx]
 
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Confidence", f"{best_prob:.1%}")
                st.progress(float(best_prob))
            with col2:
                if best_prob > 0.6:
                    st.success("🪐 EXOPLANET DETECTED!")
                    st.caption(f"Transit signal found at flux position {best_pos}")
                    st.balloons()
                else:
                    st.info("🌠 No Exoplanet Detected")
 
            with st.expander("🔍 Window Analysis"):
                st.line_chart(probabilities)
                st.caption(f"Analyzed {len(probabilities)} windows of size {window_size}")
                st.subheader("Full Light Curve")
                st.line_chart(data)
                st.caption(f"Best window: position {best_pos} → {best_pos + window_size}")
 
        else:
            # Exactly 3197 values
            st.subheader("📈 Light Curve")
            st.line_chart(data)
 
            data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)
            x_input = data_norm.reshape(1, -1, 1)
            prob = model_cnn.predict(x_input, verbose=0)[0][0]
 
            st.markdown("---")
            st.subheader("🔭 Prediction")
            st.progress(float(prob))
 
            if prob > 0.6:
                st.success(f"🪐 EXOPLANET DETECTED! Confidence: {prob:.2%}")
                st.balloons()
            else:
                st.info(f"🌠 No Exoplanet Detected. Confidence: {prob:.2%}")