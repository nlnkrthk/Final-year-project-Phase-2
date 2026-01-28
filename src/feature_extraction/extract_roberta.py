import os
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSCRIPT_ROOT = "data/transcripts/whisper"
FEATURE_ROOT = "data/features/roberta"

os.makedirs(FEATURE_ROOT, exist_ok=True)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained(
    "roberta-base",
    use_safetensors=True
)
model.to(DEVICE)
model.eval()

MAX_TOKENS = 512

def chunk_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [
        tokens[i:i + MAX_TOKENS]
        for i in range(0, len(tokens), MAX_TOKENS)
    ]
    return chunks

@torch.no_grad()
def extract_features(text):
    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        inputs = torch.tensor([chunk]).to(DEVICE)
        outputs = model(inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
        embeddings.append(cls_embedding.squeeze(0).cpu())

    return torch.stack(embeddings)  # [num_chunks, 768]

def process_class(class_name):
    in_class_dir = os.path.join(TRANSCRIPT_ROOT, class_name)
    out_class_dir = os.path.join(FEATURE_ROOT, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    for patient in tqdm(os.listdir(in_class_dir), desc=class_name):
        in_patient_dir = os.path.join(in_class_dir, patient)
        out_patient_dir = os.path.join(out_class_dir, patient)
        os.makedirs(out_patient_dir, exist_ok=True)

        for file in os.listdir(in_patient_dir):
            if not file.endswith(".txt"):
                continue

            txt_path = os.path.join(in_patient_dir, file)
            feat_path = os.path.join(
                out_patient_dir,
                file.replace(".txt", ".pt")
            )

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if len(text) == 0:
                continue

            features = extract_features(text)
            torch.save(features, feat_path)

if __name__ == "__main__":
    process_class("dementia")
    process_class("no_dementia")
