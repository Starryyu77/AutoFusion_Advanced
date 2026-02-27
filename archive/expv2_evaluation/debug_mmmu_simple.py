"""
MMMU验证集调试脚本 - 简化版
"""
import sys
sys.path.insert(0, '/Users/starryyu/2026/Auto-Fusion-Advanced')

from datasets import load_dataset, concatenate_datasets

print("Loading MMMU dataset...")
subjects = ['Accounting', 'Agriculture', 'Art', 'Biology', 'Chemistry']

# Load first subject
ds = load_dataset('MMMU/MMMU', subjects[0], split='validation', cache_dir='./data/mmmu')

print(f"\nDataset loaded: {len(ds)} samples")
print(f"\nFirst sample keys: {list(ds[0].keys())}")

# Check first 5 samples
print("\n" + "="*60)
print("Checking first 5 samples:")
for i in range(min(5, len(ds))):
    item = ds[i]
    print(f"\nSample {i}:")
    print(f"  id: {item.get('id')}")
    print(f"  question: {str(item.get('question', ''))[:80]}...")
    print(f"  options: {item.get('options', [])}")
    print(f"  answer: {item.get('answer')}")

    # Check images
    for j in range(1, 8):
        img_key = f'image_{j}'
        img = item.get(img_key)
        if img is not None:
            print(f"  image_{j}: {type(img)} - size={getattr(img, 'size', 'N/A')}")
            break
    else:
        print(f"  images: All None!")

# Test split logic
print("\n" + "="*60)
print("Testing train/val split logic:")
train_count = 0
val_count = 0
valid_samples = 0

for idx, item in enumerate(ds):
    choices = item.get('options', [])
    answer = item.get('answer', '')

    if len(choices) < 2 or len(choices) > 50:
        continue

    if isinstance(answer, str) and len(answer) > 0:
        label = ord(answer[0].upper()) - ord('A')
        if label < 0 or label >= len(choices):
            continue

    if valid_samples % 10 < 8:
        train_count += 1
    else:
        val_count += 1
    valid_samples += 1

print(f"Total valid samples: {valid_samples}")
print(f"Train: {train_count}, Val: {val_count}")

if val_count == 0:
    print("\n❌ ERROR: No validation samples!")
else:
    print(f"\n✅ Split working correctly: {train_count}/{val_count}")
