from src.models.fusion_dataset import FusionDataset

dataset = FusionDataset(
    wavlm_root="data/features/wavlm",
    roberta_root="data/features/roberta",
    split_file="data/splits/patient_splits.json",
    split="train",
    label_map={"dementia": 1, "no_dementia": 0}
)

print("Samples:", len(dataset))
x, y = dataset[0]
print("Feature shape:", x.shape)
print("Label:", y)
