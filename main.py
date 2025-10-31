import os
import numpy as np
import streamlit as st
import joblib
import librosa
import soundfile as sf
import sounddevice as sd
import wavio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

# --------------------------
# 1. Feature Extraction
# --------------------------
def extract_features(file_path):
    """Extract MFCC features from an audio file"""
    try:
        data, sample_rate = sf.read(file_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # stereo → mono
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


# --------------------------
# 2. Train or Load Model
# --------------------------
@st.cache_resource
def get_or_train_model():
    DATASET_PATH = "speech"
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "rf_emotion_model.pkl")

    emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load pretrained model if exists
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        st.info("✅ Loaded pre-trained Random Forest model.")
        return model, None

    # Else train new model
    X, y = [], []
    st.info("Training Random Forest model — please wait...")

    for dirpath, _, filenames in os.walk(DATASET_PATH):
        for file in filenames:
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) > 2:
                    emotion_code = parts[2]
                    if emotion_code in emotions:
                        feature = extract_features(os.path.join(dirpath, file))
                        if feature is not None:
                            X.append(feature)
                            y.append(emotions[emotion_code])

    if not X:
        st.error("No audio files found in 'speech/' directory.")
        return None, None

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_PATH)
    st.success("🎯 Model trained and saved successfully.")
    return model, acc


# --------------------------
# 3. Record Audio
# --------------------------
def record_audio(filename="user_input.wav", duration=4, fs=22050):
    st.info("🎙️ Recording... Speak for a few seconds.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    st.success("✅ Recording complete.")
    return filename


# --------------------------
# 4. Streamlit UI
# --------------------------
st.title("🎵 Speech Emotion Recognition")
st.markdown("Predict emotions from voice using MFCC + Random Forest")

model, acc = get_or_train_model()

if model:
    if acc:
        st.write(f"Model trained with accuracy: **{acc:.2f}**")

    st.subheader("🎤 Emotion Prediction")
    option = st.radio("Choose input method:", ["Record Audio", "Upload Audio"])

    if option == "Record Audio":
        duration = st.slider("Select duration (seconds)", 2, 10, 4)
        if st.button("Start Recording"):
            recorded_file = record_audio(duration=duration)
            st.audio(recorded_file, format="audio/wav")
            user_feat = extract_features(recorded_file)
            if user_feat is not None:
                pred = model.predict([user_feat])[0]
                st.success(f"Predicted Emotion: **{pred.upper()}**")

    elif option == "Upload Audio":
        uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
        if uploaded_file is not None:
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.audio("temp.wav", format="audio/wav")
            user_feat = extract_features("temp.wav")
            if user_feat is not None:
                pred = model.predict([user_feat])[0]
                st.success(f"Predicted Emotion: **{pred.upper()}**")
else:
    st.warning("⚠️ Model not available. Please place dataset in 'speech/' folder.")
