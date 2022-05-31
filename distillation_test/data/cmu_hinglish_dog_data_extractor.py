import numpy as np
from datasets import load_dataset

dataset = load_dataset("cmu_hinglish_dog")

splits = ["train", "test", "validation"]
split_to_data = {
    "train_source": [],
    "train_target": [],
    "test_source": [],
    "test_target": [],
    "validation_source": [],
    "validation_target": [],
}
for split in splits:
    for item in dataset[split]:
        split_to_data[split + "_source"].append(item["translation"]["en"])
        split_to_data[split + "_target"].append(item["translation"]["hi_en"])

for split in split_to_data:
    path = "cmu_hinglish_dog/" + split.replace("_", ".")
    file_to_safe = open(path, "w")
    for element in split_to_data[split]:
        file_to_safe.write(element + "\n")

file_to_safe.close()
