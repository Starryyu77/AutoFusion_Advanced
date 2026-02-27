"""
MMMU验证集调试 - 测试多科目合并
模拟服务器上的完整加载逻辑
"""
import sys
sys.path.insert(0, '/Users/starryyu/2026/Auto-Fusion-Advanced')

from datasets import load_dataset, concatenate_datasets
from experiment.data.dataset_loader import BaseDataset, custom_collate_fn
from torch.utils.data import DataLoader

print("=" * 60)
print("MMMU验证集调试 - 多科目合并")
print("=" * 60)

# 加载5个科目（模拟服务器代码）
subjects = ['Accounting', 'Agriculture', 'Art', 'Biology', 'Chemistry']
print(f"\n1. 加载5个科目: {subjects}")

print(f"\n2. 加载第一个科目: {subjects[0]}")
combined_dataset = load_dataset('MMMU/MMMU', subjects[0], split='validation', cache_dir='./data/mmmu')
print(f"   {subjects[0]}: {len(combined_dataset)} 样本")

for subject in subjects[1:]:
    print(f"   加载: {subject}")
    try:
        subject_dataset = load_dataset('MMMU/MMMU', subject, split='validation', cache_dir='./data/mmmu')
        print(f"     {subject}: {len(subject_dataset)} 样本")
        combined_dataset = concatenate_datasets([combined_dataset, subject_dataset])
    except Exception as e:
        print(f"     失败: {e}")

print(f"\n3. 合并后总样本: {len(combined_dataset)}")

# 处理数据（模拟 _load_mmmu）
print("\n4. 处理合并数据...")
train_data = []
val_data = []
valid_samples = 0
skipped_samples = 0

for idx, item in enumerate(combined_dataset):
    # 获取图片
    image = None
    for i in range(1, 8):
        img = item.get(f'image_{i}')
        if img is not None:
            image = img
            break

    choices = item.get('options', [])
    answer = item.get('answer', '')

    # 过滤条件
    if len(choices) < 2 or len(choices) > 50:
        skipped_samples += 1
        continue

    # 转换答案
    label = 0
    if isinstance(answer, str) and len(answer) > 0:
        label = ord(answer[0].upper()) - ord('A')
        if label < 0 or label >= len(choices):
            skipped_samples += 1
            continue

    processed = {
        'image': image,
        'decoded_image': image,
        'question': item.get('question'),
        'choices': choices,
        'answer': answer,
        'label': label,
        'subject': item.get('id', '').split('_')[0] if item.get('id') else 'unknown'
    }

    # 80/20分割
    if valid_samples % 10 < 8:
        train_data.append(processed)
    else:
        val_data.append(processed)
    valid_samples += 1

print(f"   有效样本: {valid_samples}")
print(f"   跳过样本: {skipped_samples}")
print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

# 检查验证集
print("\n5. 检查验证集样本...")
if val_data:
    for i, item in enumerate(val_data[:5]):
        print(f"   Val {i}: subject={item['subject']}, label={item['label']}, "
              f"image={'not None' if item['image'] else 'None'}")
else:
    print("   ❌ 验证集为空!")

# 测试DataLoader
print("\n6. 测试DataLoader...")
if val_data:
    val_dataset = BaseDataset(val_data, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    batch_count = 0
    for batch in val_loader:
        batch_count += 1
        labels = batch.get('label', [])
        print(f"   批次 {batch_count}: {len(labels)} 样本")

    print(f"\n   总批次数: {batch_count}")
    if batch_count == 0:
        print("   ❌ 没有有效批次!")
    else:
        print(f"   ✅ DataLoader工作正常")

print("\n" + "=" * 60)
