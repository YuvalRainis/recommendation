import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import itertools
import os
import os

# === Recommendation Logic ===
def recommend_best_parameters(patient_profile, model, batch_size=50000):
    pulse_limits_max = {1: 150, 2: 300, 3: 300, 4: 350, 5: 350, 7: 400}
    pulse_limits_min = {1: 50, 2: 40, 3: 60, 4: 100, 5: 200, 7: 200}
    treated_area = int(patient_profile['TreatedArea'])
    max_pulses = pulse_limits_max.get(treated_area, 1000)
    min_pulses = pulse_limits_min.get(treated_area, 1)

    if min_pulses > max_pulses:
        min_pulses = max_pulses

    valid_applicators = (
        [1] if treated_area == 1 else
        [2] if treated_area in [2, 3, 4] else
        [3]
    )

    energies = np.round(np.arange(3.0, 5.1, 0.2), 1)
    coolings = [1, 2, 3]
    passes_options = [2, 3, 4]
    pulse_range = range(min_pulses, max_pulses + 1)

    cols = ['Age', 'Gender', 'Race', 'Ethnicity', 'TreatedArea',
            'Applicator', 'Number of pulses', 'Number of passes',
            'Energy_J', 'Post_cooling_sec', 'Skin Type', 'Severity_Normalized', 'Indication']
    best_row, best_score = None, -np.inf

    def combos_generator():
        for num_pulses, num_passes, energy, cooling in itertools.product(
            pulse_range, passes_options, energies, coolings
        ):
            for applicator in valid_applicators:
                yield {
                    'Age': patient_profile['Age'],
                    'Gender': patient_profile['Gender'],
                    'Race': patient_profile['Race'],
                    'Ethnicity': patient_profile['Ethnicity'],
                    'Skin Type': patient_profile['Skin Type'],
                    'TreatedArea': treated_area,
                    'Applicator': applicator,
                    'Number of pulses': num_pulses,
                    'Number of passes': num_passes,
                    'Energy_J': energy,
                    'Post_cooling_sec': cooling,
                    'Severity_Normalized': patient_profile.get('Severity_Normalized', 0),
                    'Indication': patient_profile.get('Indication', 'Unknown')
                }

    batch = []
    for row in combos_generator():
        batch.append(row)
        if len(batch) >= batch_size:
            df_b = pd.DataFrame(batch, columns=cols)
            # ensure Indication is categorical to match training dtype
            if 'Indication' in df_b.columns:
                df_b['Indication'] = df_b['Indication'].astype('category')
            preds = model.predict(df_b[cols])
            idx = int(np.argmax(preds))
            if preds[idx] > best_score:
                best_score = float(preds[idx])
                best_row = df_b.iloc[idx].copy()
                best_row['predicted_score'] = best_score
            batch.clear()

    if batch:
        df_b = pd.DataFrame(batch, columns=cols)
        # ensure Indication is categorical to match training dtype
        if 'Indication' in df_b.columns:
            df_b['Indication'] = df_b['Indication'].astype('category')
        preds = model.predict(df_b[cols])
        idx = int(np.argmax(preds))
        if preds[idx] > best_score:
            best_score = float(preds[idx])
            best_row = df_b.iloc[idx].copy()
            best_row['predicted_score'] = best_score

    return best_row


# === Streamlit UI ===
st.title("⭐ Sofwave Recommendation Tool")

st.write("Enter the patient’s details to receive personalized treatment recommendations:")

age = st.number_input("Age", min_value=18, max_value=90, value=40)
gender = st.selectbox("Gender", [("1", "Female"), ("2", "Male")], format_func=lambda x: x[1])
race = st.selectbox("Race", [("1", "Caucasian"), ("2", "Asian"), ("3", "Black"), ("4", "Other")], format_func=lambda x: x[1])
ethnicity = st.selectbox("Ethnicity", [("1", "Non-Hispanic"), ("2", "Hispanic")], format_func=lambda x: x[1])
treated_area = st.selectbox(
    "Treated Area",
    [("1", "Forehead"), ("2", "Cheeks"), ("3", "Neck"), ("4", "Cheeks, Submental & Upper neck"), ("5", "Arms"), ("7", "Buttocks")],
    format_func=lambda x: x[1]
)
skin_type = st.number_input("Skin Type (1-6)", min_value=1, max_value=6, value=3)

# --- Severity input section ---
# --- Indication & Severity input section ---
# Allow per-indication scales and normalization. Steps reflect real-world integer scales,
# but decimal averages are accepted and normalized.
indication = st.selectbox(
    "Indication",
    [
        "Wrinkles (1-9)",
        "Acne scars (0-4)",
        "Upper arm laxity (0-4)",
        "Cellulite (0-5)",
        "Other (0-10)"
    ],
)

# Map indication -> (min, max, step)
indication_scales = {
    "Wrinkles (1-9)": (1.0, 9.0, 1.0),
    "Acne scars (0-4)": (0.0, 4.0, 1.0),
    "Upper arm laxity (0-4)": (0.0, 4.0, 1.0),
    "Cellulite (0-5)": (0.0, 5.0, 1.0),
    "Other (0-10)": (0.0, 10.0, 0.1)
}

scale_min, scale_max, scale_step = indication_scales.get(indication, (0.0, 9.0, 0.1))

# Default severity value: midpoint
default_sev = float((scale_min + scale_max) / 2.0)

# Use a number_input with appropriate step and bounds
severity_value = st.number_input(
    f"Severity Pre treatment ({indication})",
    min_value=float(scale_min),
    max_value=float(scale_max),
    value=default_sev,
    step=float(scale_step)
)

severity_normalized = None
try:
    denom = (scale_max - scale_min) if (scale_max - scale_min) != 0 else 1.0
    severity_normalized = (float(severity_value) - scale_min) / denom
    severity_normalized = max(0.0, min(1.0, severity_normalized))
except Exception:
    severity_normalized = None

if st.button("Generate Recommendation"):
    st.write("⏳ Loading model and generating recommendation...")
    try:
        model = xgb.XGBRegressor()
        # prefer a model placed in the workspace root (one level up), fall back to local
        possible_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'best_xgb_model.json')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'best_xgb_model.json')),
            os.path.abspath('best_xgb_model.json')
        ]
        loaded = False
        for p in possible_paths:
            if os.path.exists(p):
                model.load_model(p)
                loaded = True
                break
        if not loaded:
            # last resort, try default name (may raise)
            model.load_model("best_xgb_model.json")

        profile = {
            'Age': age,
            'Gender': int(gender[0]),
            'Race': int(race[0]),
            'Ethnicity': int(ethnicity[0]),
            'TreatedArea': int(treated_area[0]),
            'Skin Type': skin_type
        }

        # include normalized severity in profile if available
        if severity_normalized is not None:
            profile['Severity_Normalized'] = float(severity_normalized)

        # include indication string for downstream logic (optional)
        profile['Indication'] = indication

        result = recommend_best_parameters(profile, model)
        if result is None:
            st.error("No valid recommendation found.")
        else:
            st.success("✅ Recommended Treatment Parameters:")

            # Styling for better readability
            st.markdown("""
                <style>
                .dataframe th, .dataframe td {
                    white-space: nowrap;
                    text-align: center;
                    padding: 8px 14px;
                    font-size: 16px;
                }
                </style>
                """, unsafe_allow_html=True)

            # Display table with full width and clean index
            # result is a Series; add human-friendly applicator name and energy calculations
            res = result.copy()
            applicator_map = {1: ("Precise", 3), 2: ("Lift", 7), 3: ("LiftHD", 7)}
            app_id = int(res.get('Applicator', 0))
            app_name, pzt_count = applicator_map.get(app_id, (f"Applicator {app_id}", 1))
            res['Applicator_name'] = app_name

            # compute recommended total energy and number of squares (rounded)
            try:
                pulses = float(res.get('Number of pulses', 0))
                energy_j = float(res.get('Energy_J', 0))
                recommended_total_energy = pulses * energy_j * float(pzt_count)
                recommended_num_squares = int(round(recommended_total_energy / 750.0))
            except Exception:
                recommended_total_energy = np.nan
                recommended_num_squares = np.nan

            res['Recommended_total_energy_J'] = recommended_total_energy
            res['Recommended_num_squares'] = recommended_num_squares

            df_display = pd.DataFrame([res]).reset_index(drop=True)
            # Hide raw normalized severity if you prefer internal-only
            if 'Severity_Normalized' in df_display.columns:
                df_display = df_display.drop(columns=['Severity_Normalized'])
            st.dataframe(df_display, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

