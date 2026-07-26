from datasets import load_dataset
import pandas as pd

def load_data():
    """Wczytuje dataset clickbait_detection_dataset z HuggingFace i zwraca train/val/test jako pandas DataFrame"""
    ds = load_dataset("christinacdl/clickbait_detection_dataset")

    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()

    return train, val, test

if __name__ == "__main__":
    train, val, test = load_data()
    print(f"train: {train.shape}, val: {val.shape}, test: {test.shape}")
