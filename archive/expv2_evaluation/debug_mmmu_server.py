#!/usr/bin/env python3
"""
服务器端MMMU调试脚本
用于在ntu-gpu43上运行诊断MMMU问题
"""
import sys
sys.path.insert(0, '/usr1/home/s125mdg43_10/AutoFusion_Advanced')

from experiment.data.dataset_loader import DatasetLoader, custom_collate_fn
from torch.utils.data import DataLoader

print("=" * 60)
print("MMMU服务器端调试")
print("=" * 60)

# 使用与E2实验相同的配置
print("\n1. 创建DatasetLoader (使用E2配置)...")
loader = DatasetLoader(
    dataset_name='mmmu',
    batch_size=32,
    num_shots=256,
    data_dir='./data',
    shot_strategy='balanced',
    seed=42,
    transform=None
)

# 直接加载原始数据
print("\n2. 加载原始数据...")
train_data, val_data = loader._load_raw_data()
print(f"   Train: {len(train_data)} 样本")
print(f"   Val: {len(val_data)} 样本")

if len(val_data) == 0:
    print("\n   ❌ 验证集为空!")
    print("   检查 _load_mmmu 返回值...")
    sys.exit(1)

# 检查验证集内容
print("\n3. 检查验证集样本...")
for i in range(min(3, len(val_data))):
    item = val_data[i]
    img = item.get('image')
    decoded = item.get('decoded_image')
    print(f"   Val[{i}]: image={type(img).__name__}, decoded_image={type(decoded).__name__}")

# 使用DatasetLoader.load()创建DataLoader (模拟E2实验)
print("\n4. 使用DatasetLoader.load()创建DataLoader...")
train_loader, val_loader = loader.load()
print(f"   Val loader created")

# 检查批次
print("\n5. 检查DataLoader批次...")
batch_count = 0
empty_count = 0

for batch in val_loader:
    if not batch or len(batch.get('label', [])) == 0:
        empty_count += 1
    else:
        batch_count += 1
        if batch_count <= 3:
            print(f"   批次 {batch_count}: {len(batch.get('label', []))} 样本")

print(f"\n   总批次: {batch_count}, 空批次: {empty_count}")

if batch_count == 0:
    print("\n   ❌ 没有有效批次!")
    print("\n   诊断建议:")
    print("   1. dataset_loader.py中val_loader的drop_last仍为True")
    print("   2. 需要修复dataset_loader.py第268行")
else:
    print(f"\n   ✅ DataLoader正常: {batch_count} 个有效批次")

print("\n" + "=" * 60)
