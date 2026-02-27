"""
MMMU验证集调试脚本
诊断为什么MMMU验证集为空
"""

import sys
sys.path.insert(0, '/Users/starryyu/2026/Auto-Fusion-Advanced')

from experiment.data.dataset_loader import DatasetLoader, custom_collate_fn
from torch.utils.data import DataLoader

def debug_mmmu():
    print("=" * 60)
    print("MMMU数据集调试")
    print("=" * 60)

    # 创建加载器
    loader = DatasetLoader(
        dataset_name='mmmu',
        batch_size=4,
        num_shots=256,  # 使用完整训练模式
        data_dir='./data',
        shot_strategy='balanced',
        seed=42
    )

    # 加载原始数据
    print("\n1. 加载原始数据...")
    train_data, val_data = loader._load_raw_data()
    print(f"   原始训练集: {len(train_data)} 样本")
    print(f"   原始验证集: {len(val_data)} 样本")

    # 检查数据格式
    if train_data:
        print("\n2. 检查训练集样本格式:")
        sample = train_data[0]
        for key, value in sample.items():
            if key == 'image' or key == 'decoded_image':
                print(f"   {key}: {type(value)} {'(not None)' if value is not None else '(None)'}")
            else:
                print(f"   {key}: {type(value)} = {value if not isinstance(value, list) else value[:3]}...")

    # 检查验证集样本
    if val_data:
        print("\n3. 检查验证集样本格式:")
        sample = val_data[0]
        for key, value in sample.items():
            if key == 'image' or key == 'decoded_image':
                print(f"   {key}: {type(value)} {'(not None)' if value is not None else '(None)'}")
            else:
                print(f"   {key}: {type(value)} = {value if not isinstance(value, list) else value[:3]}...")

    # 检查custom_collate_fn过滤后的结果
    print("\n4. 检查custom_collate_fn过滤效果...")

    # 创建Dataset
    from experiment.data.dataset_loader import BaseDataset
    val_dataset = BaseDataset(val_data, transform=None)

    # 手动检查几个样本
    print(f"   验证集总样本数: {len(val_dataset)}")

    none_image_count = 0
    for i in range(min(10, len(val_dataset))):
        item = val_dataset[i]
        if item.get('image') is None and item.get('decoded_image') is None:
            none_image_count += 1
            print(f"   样本 {i}: image=None, decoded_image=None")
        else:
            img = item.get('image') or item.get('decoded_image')
            print(f"   样本 {i}: image type = {type(img)}")

    print(f"\n   前10个样本中None图像数量: {none_image_count}")

    # 尝试创建DataLoader并检查批次
    print("\n5. 创建DataLoader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    print(f"   DataLoader创建成功")

    # 尝试获取一个批次
    print("\n6. 尝试获取验证集批次...")
    batch_count = 0
    empty_batch_count = 0

    for batch in val_loader:
        if not batch or len(batch.get('label', [])) == 0:
            empty_batch_count += 1
        else:
            batch_count += 1
            if batch_count <= 3:
                print(f"   批次 {batch_count}: 大小 = {len(batch.get('label', []))}")

    print(f"\n   总批次数: {batch_count}")
    print(f"   空批次数: {empty_batch_count}")

    if batch_count == 0:
        print("\n❌ 问题确认: 验证集所有批次都为空!")
        print("可能原因:")
        print("  - custom_collate_fn过滤了所有None图像样本")
        print("  - BaseDataset.__getitem__没有正确处理MMMU的图像格式")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    debug_mmmu()
