"""
调试数据加载器，检查返回的数据格式
"""
import sys
sys.path.append('/usr1/home/s125mdg43_10/AutoFusion_Advanced/expv2/shared')
sys.path.append('/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment')

import torch
from data.dataset_loader import get_dataset_loader
from torchvision import transforms
from PIL import Image

# 获取CLIP预处理transform
transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

print("加载数据...")
loader = get_dataset_loader(
    dataset_name='ai2d',
    batch_size=4,
    num_shots=16,
    data_dir='./data',
    transform=transform
)

train_loader, test_loader = loader.load()

print("\n检查一个batch:")
for batch in train_loader:
    print(f"Batch type: {type(batch)}")
    if isinstance(batch, dict):
        for key, value in batch.items():
            print(f"  {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
    break

print("\n检查设备:")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print("\n加载CLIP...")
import clip
model, _ = clip.load('ViT-L/14', device=device)

print("\n测试编码...")
for batch in train_loader:
    if isinstance(batch, dict):
        images = batch['image']
        texts = batch.get('question') or batch.get('text')

        print(f"Images type: {type(images)}")
        print(f"Texts type: {type(texts)}")

        # 处理images
        if isinstance(images, list):
            print("Images is a list of PIL images, preprocessing...")
            images = torch.stack([transform(img) for img in images])
        if images.device != device:
            images = images.to(device)

        print(f"Images shape: {images.shape}")

        # 处理texts
        if isinstance(texts, list):
            print("Texts is a list, tokenizing...")
            texts = clip.tokenize(texts, truncate=True).to(device)

        print(f"Texts shape: {texts.shape}")

        # 测试编码
        with torch.no_grad():
            vision_features = model.encode_image(images).float()
            text_features = model.encode_text(texts).float()

        print(f"Vision features shape: {vision_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print("✓ 编码成功!")
        break
