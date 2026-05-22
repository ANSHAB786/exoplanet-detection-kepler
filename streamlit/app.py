import tensorflow
import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

st.set_page_config("Exoplanet Hunter", layout='wide')

st.title("🔭Exoplanet Hunt")
st.write("🪐Got a light curve ? Lets check for planets")
st.caption("Upload your raw light curve -- To check if their exist a planet")

@st.cache_resource # It will make the model load once and make it stays inside memory, and make it faster 
def load_exoplanet_model():
    with st.spinner("🔁Loading model"):
        return load_model('Exoplanet.best.keras')
model_cnn = load_exoplanet_model() # Model gets loaded here
st.success("✔️ Model ready!")
uploaded = st.file_uploader(
    "Choose the CSV file",
      type=['csv', 'zip'])

if uploaded:
    file_decode = ['cp1252', 'utf-8',
                    'latin1', 'iso-8859-1',
                      'utf-16']
    df = None
    used_encoding = None
    for encoding in file_decode:
        try:
            df = pd.read_csv(uploaded, encoding=encoding)
            used_encoding = encoding
            break
        except:
            continue
    if df is None:
        st.error("Could not read file. Please check the format.")
        st.stop()
    st.caption(f"File loaded (encoding): {used_encoding}")
    # we did this so that when user upload the CSV file it would not take the label or id like columns which could create noise in prediction
    numeric_df = df.select_dtypes(include=[np.number])
    # If user upload just one star
    if len(numeric_df) > 1:
        st.info(f"Processing {len(df)} stars...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        for i in range(len(numeric_df)):
            status_text.text(f"Analyzing star {i+1} of {len(numeric_df)}...")
            progress_bar.progress((i+1) / len(numeric_df))
            data = numeric_df.iloc[i].values[:3197]
            # Normalizing
            data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)
            # reshaping here because our model takes 3D input
            x_input = data_norm.reshape(1, -1, 1)
            prob = model_cnn.predict(x_input, verbose = 0)[0][0]
            results.append({
                'Star' : i+1,
                'Probability': f"{prob:.1%}",
                'Result': '🪐PLANET' if prob > 0.6 else 'NO PLANET'})
        status_text.text("✔️Analysis Complete")
        result_df = pd.DataFrame(results)
        planet_df = result_df[result_df['Result'] == '🪐PLANET']
        total_planets = len(planet_df)
        st.markdown("---")
        st.subheader("📊SUMMARY")
        col1_sum, col2_sum, col3_sum = st.columns(3)
        with col1_sum:
            st.metric("Total Stars", len(results))
        with col2_sum:
            st.metric("PLANETS FOUND", total_planets)
        with col3_sum:
            st.metric("Non-planets", len(results) - total_planets)
        if total_planets > 0:
            st.markdown("---")
            st.subheader(f"PLANET CANDITATES ({total_planets} found)")
            st.dataframe(planet_df)        # Convert them to dataframe and produce a csv file for the user so they can download it
        csv = pd.DataFrame(results).to_csv(index=False)
        st.download_button("📥 Download Results", csv, "results.csv", key="download_results")
    else:
        data = numeric_df.iloc[0].values
        
        # Here we used Sliding Window approach to handle if someone has more then 3197 feature to tackle its one limitation
        if len(data) > 3197:
            st.info(f"📊Longer light curves: {len(data)} values. Using Sliding window...")
            window_size = 3197
            step_size = 300
            probabilities = []
            positions = []

            for start in range(0, len(data) - window_size + 1, step_size):
                window = data[start: start + window_size]
                window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
                x_input = window_norm.reshape(1, -1, 1)
                prob = model_cnn.predict(x_input, verbose = 0)[0][0]
                probabilities.append(prob)
                positions.append(start)
                
                
                best_idx = np.argmax(probabilities)
                best_prob = probabilities[best_idx]
                best_pos = positions[best_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best confidence", f"{best_prob:.1%}")
                    st.progress(best_prob)
                with col2:
                    if best_prob > 0.6:
                        st.success(f"EXOPLANET DETECTED!")
                        st.caption(f" found at position {best_pos}")
                        st.balloons()
                    else:
                        st.info(f"No Exoplanet Detected")
                with st.expander("Window Analysis"):
                    st.line_chart(probabilities)
                    st.caption(f"Analyzed {len(probabilities)} windows")
                    st.subheader("Full light Curve")
                    st.line_chart(data)
                    st.caption(f"Best window position {best_pos} to {best_pos + window_size}")

            else:
                if len(data) < 3197:
                    st.error(f"❌ Need atleast 3197 values. your file has len({data})values.")
                    st.stop() # To stop the execution
                else:
                    data_norm = data

                # Normalize it to help our model gain insight
                final_norm = (data_norm - np.mean(data_norm)) / np.std(data_norm) # data gets normalized
                x_input = final_norm.reshape(1, -1, 1) # The user input gets converted in 3D
    
                prob = model_cnn.predict(x_input, verbose = 0)[0][0]

                if prob > 0.6:
                    st.success(f"🪐EXOPLANET DETECTED! with ({prob:.2%}) confidence score")
                else:
                    st.info(f"🌠NO EXOPLANET DETECTED! with ({prob:.2%}) confidence score")
                    st.line_chart()        
        else:
            st.error("No numeric data found in file")
if st.sidebar.button("Clear all"):
    st.cache_data.clear()