import os
import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, wavlm_root, roberta_root, split_file, split, label_map):
        self.samples = []

        import json
        with open(split_file, "r") as f:
            splits = json.load(f)

        for label_name, label_value in label_map.items():
            patients = splits[label_name][split]

            for patient in patients:
                wavlm_dir = os.path.join(wavlm_root, label_name, patient)
                roberta_dir = os.path.join(roberta_root, label_name, patient)

                if not os.path.exists(wavlm_dir):
                    continue

                for file in os.listdir(wavlm_dir):
                    if not file.endswith(".pt"):
                        continue

                    wavlm_path = os.path.join(wavlm_dir, file)
                    roberta_path = os.path.join(
                        roberta_dir,
                        file.replace(".pt", ".pt")
                    )

                    if not os.path.exists(roberta_path):
                        continue

                    self.samples.append(
                        (wavlm_path, roberta_path, label_value)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wavlm_path, roberta_path, label = self.samples[idx]

        wavlm_feat = torch.load(wavlm_path, weights_only=True)
        roberta_feat = torch.load(roberta_path, weights_only=True)


        wavlm_pooled = wavlm_feat.mean(dim=0)
        roberta_pooled = roberta_feat.mean(dim=0)

        fused = torch.cat([wavlm_pooled, roberta_pooled], dim=0)

        return fused, label
