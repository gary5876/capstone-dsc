"""02/03 image 노트북 PILOT_MODE 제거 + pretrained + 5-level grid + 112 resize."""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# =================================================================
# 02 노트북 cell 2 — PILOT_MODE 제거, 5-level grid
# =================================================================
NB02 = NB_DIR / "02_pollution_and_dsc_image.ipynb"
nb = json.loads(NB02.read_text(encoding="utf-8"))

NEW_02_CELL2 = """# ============================================================
# 0-2. 사전등록 (DATASETS, POLLUTION_LEVELS)
# ============================================================
# ADR-014 사전등록 원본: dataset 3종 (CIFAR10/FashionMNIST/Flowers102), level 6단계.
# Phase 1 정식 run: dataset 2종 (튜닝=CIFAR10 / held-out=FashionMNIST), level 5단계.
# 사전등록 원본 setting은 DATASETS_FULL/POLLUTION_LEVELS_FULL로 보존.

DATASETS_FULL = {
    'CIFAR10': {'loader': 'CIFAR10', 'n_classes': 10, 'image_size': 32, 'channels': 3},
    'FashionMNIST': {'loader': 'FashionMNIST', 'n_classes': 10, 'image_size': 28, 'channels': 1},
    'Flowers102': {'loader': 'Flowers102', 'n_classes': 102, 'image_size': 224, 'channels': 3},
}
POLLUTION_LEVELS_FULL = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

DATASETS = {k: v for k, v in DATASETS_FULL.items() if k in ('CIFAR10', 'FashionMNIST')}
POLLUTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
print(f'Phase 1: datasets={list(DATASETS.keys())}, levels={POLLUTION_LEVELS}')

RANDOM_SEED = 42
SAMPLE_CAP = 5000  # 폴루션 후 DSC 계산 시 sample_cap

def load_train(ds_name):
    if ds_name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(f'{DATA_DIR}/CIFAR10', train=True, download=True)
    if ds_name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(f'{DATA_DIR}/FashionMNIST', train=True, download=True)
    if ds_name == 'Flowers102':
        return torchvision.datasets.Flowers102(f'{DATA_DIR}/Flowers102', split='train', download=True)

print(f'데이터셋: {list(DATASETS.keys())}, 강도: {POLLUTION_LEVELS}')
"""

# splitlines(keepends=True) 형태로 저장 (Jupyter 관습)
nb["cells"][2]["source"] = NEW_02_CELL2.splitlines(keepends=True)
nb["cells"][2]["outputs"] = []
nb["cells"][2]["execution_count"] = None
NB02.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"02 노트북 cell 2 갱신 완료: {NB02}")

# =================================================================
# 03 노트북 — cell 2 (DATASETS + EPOCHS + pretrained) + cell 6 (image_size)
# =================================================================
NB03 = NB_DIR / "03_training_image.ipynb"
nb = json.loads(NB03.read_text(encoding="utf-8"))

NEW_03_CELL2 = """# ============================================================
# 0-2. 사전등록 + 모델 정의
# ============================================================
# ADR-014 사전등록 원본: dataset 3종, 모델 5종, EPOCHS=30, level 6단계.
# Phase 1 정식 run: dataset 2종, 모델 2종 (ResNet18 pretrained + CNNSimple),
#                    EPOCHS=10, level 5단계. image_size는 dataset별 분기.
# 사전등록 원본 setting은 *_FULL로 보존.

DATASETS_FULL = {
    'CIFAR10': {'n_classes': 10, 'image_size': 112, 'channels': 3},
    'FashionMNIST': {'n_classes': 10, 'image_size': 112, 'channels': 1},
    'Flowers102': {'n_classes': 102, 'image_size': 224, 'channels': 3},
}
MODEL_NAMES_FULL = ['ResNet18', 'EfficientNetB0', 'MobileNetV3small', 'ViTTiny', 'CNNSimple']
EPOCHS_FULL = 30  # ADR-014 사전등록
POLLUTION_LEVELS_FULL = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

DATASETS = {k: v for k, v in DATASETS_FULL.items() if k in ('CIFAR10', 'FashionMNIST')}
MODEL_NAMES = ['ResNet18', 'CNNSimple']
EPOCHS = 10
POLLUTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
print(f'Phase 1: datasets={list(DATASETS.keys())}, models={MODEL_NAMES}, EPOCHS={EPOCHS}, levels={POLLUTION_LEVELS}')

BATCH_SIZE = 128
LR = 1e-3


def get_model(model_name, n_classes, in_channels=3):
    if model_name == 'ResNet18':
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    if model_name == 'EfficientNetB0':
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        return m
    if model_name == 'MobileNetV3small':
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        return m
    if model_name == 'ViTTiny':
        import timm
        return timm.create_model('vit_tiny_patch16_224', pretrained=True,
                                 num_classes=n_classes, in_chans=in_channels)
    if model_name == 'CNNSimple':
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

print('모델 정의 완료')
"""
nb["cells"][2]["source"] = NEW_03_CELL2.splitlines(keepends=True)
nb["cells"][2]["outputs"] = []
nb["cells"][2]["execution_count"] = None

# cell 6: get_transform(image_size=224) → meta['image_size']
old6 = "".join(nb["cells"][6]["source"]) if isinstance(nb["cells"][6]["source"], list) else nb["cells"][6]["source"]
new6 = old6.replace(
    "transform = get_transform(image_size=224)",
    "transform = get_transform(image_size=meta['image_size'])",
)
assert new6 != old6, "03 cell 6 image_size patch 실패 — 원문 매칭 안 됨"
nb["cells"][6]["source"] = new6.splitlines(keepends=True)
nb["cells"][6]["outputs"] = []
nb["cells"][6]["execution_count"] = None

NB03.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"03 노트북 cell 2/6 갱신 완료: {NB03}")
