from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

print("Downloading DBpedia-14 dataset...")
dataset = load_dataset("dbpedia_14")

print("Saving train.csv...")
dataset["train"].to_csv("data/train.csv", index=False)

print("Saving test.csv...")
dataset["test"].to_csv("data/test.csv", index=False)

print("Dataset downloaded successfully!")