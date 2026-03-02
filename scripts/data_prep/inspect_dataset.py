
from datasets import load_dataset

try:
    print("Loading first row of argilla/ultrafeedback-binarized-preferences-cleaned...")
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train", streaming=True)
    for row in dataset:
        print("Keys:", row.keys())
        print("Sample:", row)
        break
except Exception as e:
    print(f"Error: {e}")
