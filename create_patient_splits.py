import os
import random
import json

random.seed(42)

DATASET_ROOT = "."   # change if needed
OUTPUT_DIR = "data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_patients(class_name, train_ratio=0.7, val_ratio=0.15):
    class_path = os.path.join(DATASET_ROOT, class_name)
    patients = [p for p in os.listdir(class_path)
                if os.path.isdir(os.path.join(class_path, p))]

    random.shuffle(patients)

    n = len(patients)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train = patients[:n_train]
    val = patients[n_train:n_train + n_val]
    test = patients[n_train + n_val:]

    return train, val, test

splits = {}

for cls in ["dementia", "no_dementia"]:
    train, val, test = split_patients(cls)
    splits[cls] = {
        "train": train,
        "val": val,
        "test": test
    }

with open(os.path.join(OUTPUT_DIR, "patient_splits.json"), "w") as f:
    json.dump(splits, f, indent=2)

print("Split summary:")
for cls in splits:
    print(f"\nClass: {cls}")
    for split in splits[cls]:
        print(f"  {split}: {len(splits[cls][split])} patients")
