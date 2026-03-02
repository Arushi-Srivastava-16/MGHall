
from datasets import load_dataset
from tqdm import tqdm

try:
    print("Scanning for model names...")
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train", streaming=True)
    
    models = set()
    for i, row in enumerate(dataset):
        if row.get('chosen-model'): models.add(row['chosen-model'])
        if row.get('rejected-model'): models.add(row['rejected-model'])
        if i > 2000: break
    
    print("\nUnique Models Found:")
    for m in sorted(list(models)):
        print(f"  - {m}")
        
except Exception as e:
    print(f"Error: {e}")
