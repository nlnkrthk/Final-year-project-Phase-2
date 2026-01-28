import os
import librosa
import soundfile as sf
from tqdm import tqdm

RAW_ROOT = "."
OUT_ROOT = "data/processed"

TARGET_SR = 16000
TOP_DB = 25  # conservative silence trimming

os.makedirs(OUT_ROOT, exist_ok=True)

def preprocess_file(in_path, out_path):
    # Load audio
    y, sr = librosa.load(in_path, sr=None, mono=True)

    # Resample if needed
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # Trim only leading & trailing silence
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # Peak normalization
    if y.max() > 0:
        y = y / abs(y).max()

    # Save
    sf.write(out_path, y, TARGET_SR)

def process_class(class_name):
    in_class_dir = os.path.join(RAW_ROOT, class_name)
    out_class_dir = os.path.join(OUT_ROOT, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    for patient in tqdm(os.listdir(in_class_dir), desc=class_name):
        in_patient_dir = os.path.join(in_class_dir, patient)
        out_patient_dir = os.path.join(out_class_dir, patient)
        os.makedirs(out_patient_dir, exist_ok=True)

        for file in os.listdir(in_patient_dir):
            if not file.endswith(".wav"):
                continue

            in_path = os.path.join(in_patient_dir, file)
            out_path = os.path.join(out_patient_dir, file)

            preprocess_file(in_path, out_path)

if __name__ == "__main__":
    process_class("dementia")
    process_class("no_dementia")
