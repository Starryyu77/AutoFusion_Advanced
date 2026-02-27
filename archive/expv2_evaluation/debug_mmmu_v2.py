"""
MMMU验证集调试脚本 - 检查BaseDataset和custom_collate_fn
"""
import sys
sys.path.insert(0, '/Users/starryyu/2026/Auto-Fusion-Advanced')

from datasets import load_dataset
from experiment.data.dataset_loader import BaseDataset, custom_collate_fn
from torch.utils.data import DataLoader

print("=" * 60)
print("MMMU验证集调试 - 检查DataLoader流程")
print("=" * 60)

# 模拟 _load_mmmu 加载的数据
print("\n1. 加载MMMU Accounting数据...")
ds = load_dataset('MMMU/MMMU', 'Accounting', split='validation', cache_dir='./data/mmmu')

# 模拟处理流程 (简化版，只取5个样本)
print("\n2. 处理样本数据 (模拟 _load_mmmu)...")
raw_data = []
valid_samples = 0
for idx, item in enumerate(ds):
    # Process all samples to test split logic

    # 获取图片
    image = None
    for i in range(1, 8):
        img = item.get(f'image_{i}')
        if img is not None:
            image = img
            break

    choices = item.get('options', [])
    answer = item.get('answer', '')

    if len(choices) < 2 or len(choices) > 50:
        continue

    label = 0
    if isinstance(answer, str) and len(answer) > 0:
        label = ord(answer[0].upper()) - ord('A')
        if label < 0 or label >= len(choices):
            continue

    processed = {
        'image': image,
        'decoded_image': image,  # MMMU stores here too
        'question': item.get('question'),
        'choices': choices,
        'answer': answer,
        'label': label,
    }

    # 模拟80/20分割
    if valid_samples % 10 < 8:
        processed['split'] = 'train'
    else:
        processed['split'] = 'val'

    raw_data.append(processed)
    valid_samples += 1

print(f"   处理完成: {len(raw_data)} 个有效样本")
train_data = [d for d in raw_data if d['split'] == 'train']
val_data = [d for d in raw_data if d['split'] == 'val']
print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

# 创建BaseDataset并测试
print("\n3. 创建BaseDataset并测试 __getitem__...")
val_dataset = BaseDataset(val_data, transform=None)
print(f"   Val dataset size: {len(val_dataset)}")

# 检查每个验证样本
print("\n4. 检查验证样本 __getitem__ 结果:")
for i in range(len(val_dataset)):
    item = val_dataset[i]
    img = item.get('image')
    decoded = item.get('decoded_image')
    print(f"   样本 {i}: image type = {type(img).__name__}, decoded_image type = {type(decoded).__name__}")

# 测试 custom_collate_fn
print("\n5. 测试 custom_collate_fn...")
if len(val_dataset) > 0:
    # 创建一个小批次
    batch = [val_dataset[i] for i in range(len(val_dataset))]
    print(f"   原始批次大小: {len(batch)}")

    collated = custom_collate_fn(batch)
    print(f"   Collated结果: {type(collated)}")
    if collated:
        for key, value in collated.items():
            if hasattr(value, '__len__'):
                print(f"     {key}: {type(value).__name__} (len={len(value)})")
            else:
                print(f"     {key}: {type(value).__name__} = {value}")
    else:
        print("     ⚠️ collated为空!")

# 创建DataLoader并测试
print("\n6. 创建DataLoader并测试:")
if len(val_dataset) > 0:
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    batch_count = 0
    for batch in val_loader:
        batch_count += 1
        labels = batch.get('label', [])
        print(f"   批次 {batch_count}: {len(labels)} 个样本")

    print(f"\n   总批次数: {batch_count}")
    if batch_count == 0:
        print("   ❌ 没有有效批次!")

print("\n" + "=" * 60)
