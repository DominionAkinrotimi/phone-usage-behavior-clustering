import streamlit as st
import joblib
import numpy as np

# Feature names
features = [
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Number of Apps Installed',
    'Data Usage (MB/day)'
]

@st.cache_resource
def load_models():
    dt_model = joblib.load('decision_tree_model.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    return dt_model, rf_model

def main():
    st.title("User Behavior Class Prediction")

    dt_model, rf_model = load_models()

    st.write("Input the following features:")

    # Collect user input
    inputs = []
    inputs.append(st.number_input(features[0], min_value=0.0, value=100.0))
    inputs.append(st.number_input(features[1], min_value=0.0, value=5.0))
    inputs.append(st.number_input(features[2], min_value=0.0, value=2000.0))
    inputs.append(st.number_input(features[3], min_value=0, step=1, value=20))
    inputs.append(st.number_input(features[4], min_value=0.0, value=500.0))

    input_array = np.array(inputs).reshape(1, -1)

    if st.button("Predict"):
        # Predict probabilities
        dt_probs = dt_model.predict_proba(input_array)[0]
        rf_probs = rf_model.predict_proba(input_array)[0]

        # Average probabilities
        avg_probs = (dt_probs + rf_probs) / 2

        # Get predicted class (index of max average prob)
        pred_class_idx = np.argmax(avg_probs)
        pred_class_prob = avg_probs[pred_class_idx]

        # Map class index to label
        classes = dt_model.classes_
        pred_class = classes[pred_class_idx]

        st.subheader("Prediction Results")
        st.write(f"Predicted User Behavior Class: **{pred_class}**")
        st.write(f"Confidence Level: **{pred_class_prob:.2f}**")

        st.write("Average Class Probabilities:")
        for cls, prob in zip(classes, avg_probs):
            st.write(f"{cls}: {prob:.2f}")

if __name__ == "__main__":
    main()
