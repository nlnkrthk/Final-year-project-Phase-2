import os
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROCESSED_ROOT = "data/processed"
OUT_ROOT = "data/transcripts/whisper"

os.makedirs(OUT_ROOT, exist_ok=True)

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    use_safetensors=True
)
model.to(DEVICE)
model.eval()

@torch.no_grad()
def transcribe(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = processor(
        waveform.squeeze(0),
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription

def process_class(class_name):
    in_class_dir = os.path.join(PROCESSED_ROOT, class_name)
    out_class_dir = os.path.join(OUT_ROOT, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    for patient in tqdm(os.listdir(in_class_dir), desc=class_name):
        in_patient_dir = os.path.join(in_class_dir, patient)
        out_patient_dir = os.path.join(out_class_dir, patient)
        os.makedirs(out_patient_dir, exist_ok=True)

        for file in os.listdir(in_patient_dir):
            if not file.endswith(".wav"):
                continue

            wav_path = os.path.join(in_patient_dir, file)
            txt_path = os.path.join(
                out_patient_dir,
                file.replace(".wav", ".txt")
            )

            text = transcribe(wav_path)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

if __name__ == "__main__":
    process_class("dementia")
    process_class("no_dementia")
