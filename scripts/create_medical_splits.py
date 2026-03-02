"""Create improved medical splits for validation."""

import json
import random
from pathlib import Path

# Paths
project_root = Path(__file__).parent.parent
input_file = project_root / "data/processed/medical_synthetic_improved.jsonl"
output_dir = project_root / "data/processed/medical_improved_splits"

output_dir.mkdir(parents=True, exist_ok=True)

# Load data
chains = []
with open(input_file) as f:
    for line in f:
        chains.append(json.loads(line))

print(f"Loaded {len(chains)} chains")

# Shuffle
random.seed(42)
random.shuffle(chains)

# Split 70/15/15
n = len(chains)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

train = chains[:train_size]
val = chains[train_size:train_size + val_size]
test = chains[train_size + val_size:]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Save splits
for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
    output_path = output_dir / f"{split_name}.jsonl"
    with open(output_path, "w") as f:
        for chain in split_data:
            f.write(json.dumps(chain) + "\\n")
    print(f"Saved {split_name}: {output_path}")

print("✅ Splits created!")
