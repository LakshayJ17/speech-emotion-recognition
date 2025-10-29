import os
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import wavio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 1. Load and extract features
def extract_features(file_path):
    """Extract MFCC features from an audio file"""
    try:
        data, sample_rate = sf.read(file_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # convert stereo to mono
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None



# 2. Prepare dataset
DATASET_PATH = "speech"  

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

X, y = [], []

print("Extracting features from dataset...")

for dirpath, _, filenames in os.walk(DATASET_PATH):
    for file in filenames:
        if file.endswith(".wav"):
            file_path = os.path.join(dirpath, file)
            parts = file.split("-")
            emotion_code = parts[2]
            if emotion_code in emotions:
                feature = extract_features(file_path)
                if feature is not None:
                    X.append(feature)
                    y.append(emotions[emotion_code])

X = np.array(X)
y = np.array(y)

print(f"Feature extraction complete. Samples: {len(X)}")


# 3. Split and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Record audio from mic and predict
def record_audio(filename="user_input.wav", duration=4, fs=22050):
    """Record audio from microphone"""
    print("\n🎙️ Recording... Speak for a few seconds.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print("✅ Recording complete.\n")
    return filename


print("\n--- Emotion Prediction from Microphone ---")
recorded_file = record_audio()

user_feat = extract_features(recorded_file)
if user_feat is not None:
    prediction = model.predict([user_feat])[0]
    print(f"Predicted Emotion: {prediction.upper()}")
else:
    print("Could not extract features from recorded audio.")
