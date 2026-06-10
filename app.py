import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import itertools
import re
import os

# Helper: map numeric severity to rubric label
def map_severity_label(value, rubric="3-level", scale_min=0.0, scale_max=10.0):
    try:
        v = float(value)
    except Exception:
        return str(value)

    # protect divide by zero
    try:
        span = float(scale_max) - float(scale_min)
        if span <= 0:
            # fallback to 0-10
            scale_min, scale_max, span = 0.0, 10.0, 10.0
    except Exception:
        scale_min, scale_max, span = 0.0, 10.0, 10.0

    # normalize to 0..1
    normalized = (v - float(scale_min)) / span
    # clamp
    normalized = max(0.0, min(1.0, normalized))

    if rubric == "4-level":
        if normalized <= 0.25:
            return "Mild"
        if normalized <= 0.50:
            return "Moderate"
        if normalized <= 0.75:
            return "Severe"
        return "Extreme"
    else:
        if normalized <= 1/3:
            return "Mild"
        if normalized <= 2/3:
            return "Moderate"
        return "Severe"

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
            'Energy_J', 'Post_cooling_sec', 'Skin Type', 'Severity_Normalized']
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
                    'Severity_Normalized': patient_profile.get('Severity_Normalized', 0)
                }

    batch = []
    for row in combos_generator():
        batch.append(row)
        if len(batch) >= batch_size:
            df_b = pd.DataFrame(batch, columns=cols)
            preds = model.predict(df_b[cols])
            idx = int(np.argmax(preds))
            if preds[idx] > best_score:
                best_score = float(preds[idx])
                best_row = df_b.iloc[idx].copy()
                best_row['predicted_score'] = best_score
            batch.clear()

    if batch:
        df_b = pd.DataFrame(batch, columns=cols)
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
st.markdown("### Severity (pre-treatment)")
severity_source = st.radio("Severity input method", ["Manual entry", "Upload Excel (worksheet with severity)"], index=0)

severity_value = None
severity_normalized = None
severity_rubric = st.selectbox("Severity rubric", ["Mild / Moderate / Severe", "Mild / Moderate / Severe / Extreme"], format_func=lambda x: x)
rubric_mode = "3-level" if severity_rubric.startswith("Mild / Moderate / Severe") and "Extreme" not in severity_rubric else "4-level"

if severity_source == "Manual entry":
    # Manual entry is always on the ES 1-9 scale
    severity_value = st.number_input("Severity Pre treatment (ES, 1.0-9.0)", min_value=1.0, max_value=9.0, value=5.5, step=0.1)
    scale_min, scale_max = 1.0, 9.0
    try:
        denom = (scale_max - scale_min) if (scale_max - scale_min) != 0 else 1.0
        severity_normalized = (float(severity_value) - scale_min) / denom
        severity_normalized = max(0.0, min(1.0, severity_normalized))
    except Exception:
        severity_normalized = None

    severity_category = map_severity_label(severity_value, rubric_mode, scale_min=scale_min, scale_max=scale_max)
    st.write(f"Mapped severity: {severity_category} (normalized={severity_normalized})")
else:
    uploaded = st.file_uploader("Upload Excel file (xlsx)", type=["xlsx", "xls"]) 
    if uploaded is not None:
        try:
            df_uploaded = pd.read_excel(uploaded, sheet_name=None)
            # try to find the expected worksheet name first
            sheet_name = None
            for s in df_uploaded.keys():
                if "DB Bruto body" in s or "Bruto" in s:
                    sheet_name = s
                    break
            if sheet_name is None:
                # fall back to first sheet
                sheet_name = list(df_uploaded.keys())[0]

            df_sheet = df_uploaded[sheet_name]
            st.write(f"Loaded sheet: {sheet_name} — {df_sheet.shape[0]} rows")

            # try to detect severity columns (value and scale)
            candidates = [c for c in df_sheet.columns if "Severity" in str(c)]
            if not candidates:
                st.warning("No column with 'Severity' in the header was found. Please check your file.")
            else:
                # split value vs scale columns
                value_cols = [c for c in candidates if 'scale' not in str(c).lower()]
                scale_cols = [c for c in candidates if 'scale' in str(c).lower()]

                value_col = value_cols[0] if value_cols else candidates[0]
                if len(value_cols) > 1:
                    value_col = st.selectbox("Detected severity value columns", value_cols)

                scale_col = scale_cols[0] if scale_cols else None
                if len(scale_cols) > 1:
                    scale_col = st.selectbox("Detected severity scale columns", scale_cols)

                # let user pick row
                idx_options = df_sheet.index.tolist()
                chosen_idx = st.selectbox("Choose row (patient) to read severity from", idx_options)

                val = df_sheet.loc[chosen_idx, value_col]
                st.write(f"Value from file ({value_col}): {val}")

                # try to obtain scale text (prefer same row in scale_col, else first non-null)
                scale_min, scale_max = 0.0, 10.0
                if scale_col is not None:
                    scale_text = df_sheet.loc[chosen_idx, scale_col]
                    if pd.isna(scale_text):
                        non_null = df_sheet[scale_col].dropna()
                        if len(non_null) > 0:
                            scale_text = str(non_null.iloc[0])
                    if not pd.isna(scale_text):
                        st.write(f"Detected scale text ({scale_col}): {scale_text}")
                        # parse patterns like 'scale 1-9' or '1-9'
                        m = re.search(r"(\d+)\s*[-–]\s*(\d+)", str(scale_text))
                        if not m:
                            m = re.search(r"scale\s*(\d+)\s*[-–]\s*(\d+)", str(scale_text), flags=re.IGNORECASE)
                        if m:
                            try:
                                scale_min = float(m.group(1))
                                scale_max = float(m.group(2))
                            except Exception:
                                scale_min, scale_max = 0.0, 10.0

                try:
                    severity_value = float(val)
                except Exception:
                    severity_value = val

                # compute normalized severity based on parsed scale
                try:
                    denom = (scale_max - scale_min) if (scale_max - scale_min) != 0 else 1.0
                    severity_normalized = (float(severity_value) - scale_min) / denom
                    severity_normalized = max(0.0, min(1.0, severity_normalized))
                except Exception:
                    severity_normalized = None

                severity_category = map_severity_label(severity_value, rubric_mode, scale_min=scale_min, scale_max=scale_max)
                st.write(f"Mapped severity: {severity_category} (normalized={severity_normalized})")
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")

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
            df_display = pd.DataFrame([result]).reset_index(drop=True)
            st.dataframe(df_display, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

