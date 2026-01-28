import os
import torch
import torchaudio
from tqdm import tqdm
from transformers import WavLMModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROCESSED_ROOT = "data/processed"
FEATURE_ROOT = "data/features/wavlm"

os.makedirs(FEATURE_ROOT, exist_ok=True)

# Load pretrained WavLM
model = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus",
    use_safetensors=True
)
model.to(DEVICE)
model.eval()

@torch.no_grad()
def extract_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    # waveform shape: [1, T]
    waveform = waveform.to(DEVICE)

    outputs = model(waveform)
    features = outputs.last_hidden_state.squeeze(0).cpu()

    return features  # [frames, 768]

def process_class(class_name):
    in_class_dir = os.path.join(PROCESSED_ROOT, class_name)
    out_class_dir = os.path.join(FEATURE_ROOT, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    for patient in tqdm(os.listdir(in_class_dir), desc=class_name):
        in_patient_dir = os.path.join(in_class_dir, patient)
        out_patient_dir = os.path.join(out_class_dir, patient)
        os.makedirs(out_patient_dir, exist_ok=True)

        for file in os.listdir(in_patient_dir):
            if not file.endswith(".wav"):
                continue

            wav_path = os.path.join(in_patient_dir, file)
            feat_path = os.path.join(
                out_patient_dir,
                file.replace(".wav", ".pt")
            )

            features = extract_features(wav_path)
            torch.save(features, feat_path)

if __name__ == "__main__":
    process_class("dementia")
    process_class("no_dementia")
