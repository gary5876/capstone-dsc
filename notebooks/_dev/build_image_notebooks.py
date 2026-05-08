"""이미지 cell 노트북 4개 일괄 생성 (01_image, 02_image, 03_image, 04_image).

ADR-014 사전등록 + 이미지 cell 마스터플랜 (20260508-01) 따라:
- 데이터셋 3개: CIFAR-10, Fashion-MNIST, Flowers102 (torchvision)
- 모델 5개: ResNet-18, EfficientNet-B0, MobileNetV3-small, ViT-Tiny, CNN-Simple
- Polluter 5개: completeness_image, noise_injection, blur, class_balance, label_swap
- 평가: accuracy

Colab GPU 환경 가정 (T4/L4). PyTorch + torchvision 표준.
체크포인트: 03 노트북에서 (dataset, polluter, level, model) 단위로 skip.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


def make_nb(cells, fname):
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 4,
    }
    out = REPO / 'notebooks' / fname
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    n_md = sum(1 for c in cells if c['cell_type'] == 'markdown')
    n_code = sum(1 for c in cells if c['cell_type'] == 'code')
    print(f'{fname}: 총 {len(cells)} 셀 (md {n_md}, code {n_code})')


# =========================================================================
# 노트북 01 — Setup & Baseline (Image)
# =========================================================================

NB01 = []
NB01.append(md("""# 01. Setup & Baseline (Image Cell)

**Phase 1**: torchvision 데이터셋 로드 → DSC 베이스라인 → 모델 5개 베이스라인

DSC v5 framework — image × classification cell (ADR-014 사전등록).

---"""))

NB01.append(md("""## 0. 환경 설정"""))

NB01.append(code("""# ============================================================
# 0-1. Drive 마운트 + GPU 확인
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json
import numpy as np
import torch

BASE = '/content/drive/MyDrive/capstone/dsc'
RESULTS_DIR = f'{BASE}/results'
DATA_DIR = f'{BASE}/data/image'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if BASE not in sys.path:
    sys.path.insert(0, BASE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
print(f'torch: {torch.__version__}')"""))

NB01.append(code("""# ============================================================
# 0-2. 의존성 설치 (Colab 환경)
# ============================================================
%pip install -q timm imagehash opencv-python-headless"""))

NB01.append(md("""## 1. 데이터셋 로드 — CIFAR-10 / Fashion-MNIST / Flowers102"""))

NB01.append(code("""# ============================================================
# 1-1. 데이터셋 사전등록 메타
# ============================================================
DATASETS = {
    'CIFAR10': {
        'loader': 'torchvision.datasets.CIFAR10',
        'n_classes': 10, 'image_size': 32, 'channels': 3,
    },
    'FashionMNIST': {
        'loader': 'torchvision.datasets.FashionMNIST',
        'n_classes': 10, 'image_size': 28, 'channels': 1,
    },
    'Flowers102': {
        'loader': 'torchvision.datasets.Flowers102',
        'n_classes': 102, 'image_size': 224, 'channels': 3,
    },
}
print(f'데이터셋: {list(DATASETS.keys())}')"""))

NB01.append(code("""# ============================================================
# 1-2. 데이터셋 다운로드 (torchvision)
# ============================================================
import torchvision

def load_dataset(ds_name, train=True):
    if ds_name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(f'{DATA_DIR}/CIFAR10', train=train, download=True)
    elif ds_name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(f'{DATA_DIR}/FashionMNIST', train=train, download=True)
    elif ds_name == 'Flowers102':
        split = 'train' if train else 'test'
        return torchvision.datasets.Flowers102(f'{DATA_DIR}/Flowers102', split=split, download=True)
    raise ValueError(ds_name)

datasets_loaded = {}
for ds_name in DATASETS:
    print(f'\\n{ds_name} 로드...')
    train_ds = load_dataset(ds_name, train=True)
    test_ds = load_dataset(ds_name, train=False)
    datasets_loaded[ds_name] = (train_ds, test_ds)
    print(f'  train: {len(train_ds)}, test: {len(test_ds)}')"""))

NB01.append(code("""# ============================================================
# 1-3. PIL → numpy 추출 helper
# ============================================================
def dataset_to_arrays(ds, sample_cap=None, random_state=1):
    images, labels = [], []
    n = len(ds) if sample_cap is None else min(len(ds), sample_cap)
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(ds))[:n] if sample_cap else range(n)
    for i in idx:
        img, lbl = ds[i]
        images.append(np.array(img))
        labels.append(int(lbl))
    return images, labels

# 각 데이터셋의 train sample (DSC 계산용 — 큰 데이터는 sample_cap 적용)
SAMPLE_CAP = 5000  # DSC 메트릭 sample_cap (메모리/시간 절약)
print(f'DSC sample_cap = {SAMPLE_CAP} (이미지 수)')"""))

NB01.append(md("""## 2. DSC 베이스라인 (clean train data)"""))

NB01.append(code("""# ============================================================
# 2-1. DSC framework import + 베이스라인 계산
# ============================================================
from dsc_framework import compute_dsc_image, DEFAULT_WEIGHTS_IMAGE

print('이미지 cell DSC 엔진 import 완료')
print(f'사전등록 가중치 (sum={sum(DEFAULT_WEIGHTS_IMAGE.values()):.2f}):')
for k, v in DEFAULT_WEIGHTS_IMAGE.items():
    print(f'  {k:<35s} {v:.2f}')"""))

NB01.append(code("""# ============================================================
# 2-2. 데이터셋별 베이스라인 DSC
# ============================================================
baseline_dsc_rows = []
for ds_name, (train_ds, _) in datasets_loaded.items():
    print(f'\\n{ds_name} DSC 계산 중...')
    images, labels = dataset_to_arrays(train_ds, sample_cap=SAMPLE_CAP)
    res = compute_dsc_image(images, labels, sample_cap=SAMPLE_CAP)
    print(f'  DSC = {res[\"score\"]} ({res[\"grade\"]})')
    baseline_dsc_rows.append({'dataset': ds_name, 'polluter': 'none', 'level': 0.0, **res})

import pandas as pd
df_baseline_dsc = pd.DataFrame(baseline_dsc_rows)
df_baseline_dsc"""))

NB01.append(md("""## 3. 베이스라인 모델 학습 (5 모델 × 3 데이터셋)

GPU 시간 절약: epochs=10 sanity check, 정식 평가는 03 노트북에서 epochs=30."""))

NB01.append(code("""# ============================================================
# 3-1. 모델 정의 (사전등록)
# ============================================================
import torch.nn as nn
import torchvision.models as tvm

def get_model(model_name, n_classes, in_channels=3, image_size=32):
    if model_name == 'ResNet18':
        m = tvm.resnet18(weights=None)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    if model_name == 'EfficientNetB0':
        m = tvm.efficientnet_b0(weights=None)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        return m
    if model_name == 'MobileNetV3small':
        m = tvm.mobilenet_v3_small(weights=None)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        return m
    if model_name == 'ViTTiny':
        import timm
        return timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                 num_classes=n_classes, in_chans=in_channels)
    if model_name == 'CNNSimple':
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )
    raise ValueError(model_name)

MODEL_NAMES = ['ResNet18', 'EfficientNetB0', 'MobileNetV3small', 'ViTTiny', 'CNNSimple']
print(f'모델 5개: {MODEL_NAMES}')"""))

NB01.append(code("""# ============================================================
# 3-2. 학습 루프 (epochs=10 sanity)
# ============================================================
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_transform(ds_name, train=True):
    meta = DATASETS[ds_name]
    size = max(meta['image_size'], 224 if 'ViT' in '|'.join(MODEL_NAMES) else meta['image_size'])
    tfs = [T.Resize((size, size)), T.ToTensor()]
    if meta['channels'] == 1:
        tfs.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))  # RGB로 복제 (ResNet 등 호환)
    tfs.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(tfs)


class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base; self.transform = transform
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, lbl = self.base[i]
        if self.transform: img = self.transform(img)
        return img, lbl


def train_eval(model, train_ds, test_ds, epochs=10, batch_size=128, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
    # eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


from time import time
EPOCHS_BASELINE = 10  # sanity (정식은 03 노트북에서 30)

baseline_perf_rows = []
for ds_name, (train_raw, test_raw) in datasets_loaded.items():
    transform = get_transform(ds_name)
    train_ds = WrappedDataset(train_raw, transform)
    test_ds = WrappedDataset(test_raw, transform)
    meta = DATASETS[ds_name]

    for model_name in MODEL_NAMES:
        t0 = time()
        try:
            model = get_model(model_name, meta['n_classes'], in_channels=3,
                            image_size=meta['image_size'])
            acc = train_eval(model, train_ds, test_ds, epochs=EPOCHS_BASELINE)
        except Exception as e:
            print(f'  [{ds_name}/{model_name}] 학습 실패: {e}')
            acc = float('nan')
        elapsed = time() - t0
        baseline_perf_rows.append({
            'dataset': ds_name, 'polluter': 'none', 'level': 0.0,
            'model': model_name, 'accuracy': round(acc, 4) if not np.isnan(acc) else None,
            'epochs': EPOCHS_BASELINE,
        })
        print(f'  [{ds_name}/{model_name:<18s}] acc={acc:.4f}  ({elapsed:.0f}s)')

df_baseline_perf = pd.DataFrame(baseline_perf_rows)
df_baseline_perf"""))

NB01.append(md("""## 4. 결과 저장"""))

NB01.append(code("""# ============================================================
# 4-1. 결과 저장 (이미지 cell — 별도 파일)
# ============================================================
dsc_path = f'{RESULTS_DIR}/dsc_scores_image.csv'
perf_path = f'{RESULTS_DIR}/model_performance_image.csv'

def upsert_baseline(path, new_df):
    if os.path.isfile(path):
        existing = pd.read_csv(path)
        baseline_mask = (existing.polluter == 'none') & (existing.level == 0.0)
        kept = existing[~baseline_mask].copy()
        for col in new_df.columns:
            if col not in kept.columns:
                kept[col] = pd.NA
        extra = [c for c in kept.columns if c not in new_df.columns]
        kept = kept[list(new_df.columns) + extra]
        combined = pd.concat([kept, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return len(combined)

n1 = upsert_baseline(dsc_path, df_baseline_dsc)
n2 = upsert_baseline(perf_path, df_baseline_perf)
print(f'DSC 저장: {dsc_path} (총 {n1}건)')
print(f'모델 성능 저장: {perf_path} (총 {n2}건)')
print('--- 노트북 01 이미지 cell 완료 ---')"""))

make_nb(NB01, '01_setup_and_baseline_image.ipynb')


# =========================================================================
# 노트북 02 — Pollution & DSC (Image)
# =========================================================================

NB02 = []
NB02.append(md("""# 02. Pollution & DSC Scoring (Image Cell)

**Phase 1**: 5종 polluter × 6 level × 3 데이터셋 → DSC 점수 측정.

split-first 원칙: train/test 분할 → train에만 polluter 적용. (torchvision의 train/test split 그대로 사용 — 별도 split 불필요)

---"""))

NB02.append(code("""# ============================================================
# 0-1. 환경 + 데이터 로드
# ============================================================
from google.colab import drive; drive.mount('/content/drive')
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torchvision

BASE = '/content/drive/MyDrive/capstone/dsc'
RESULTS_DIR = f'{BASE}/results'
DATA_DIR = f'{BASE}/data/image'
POLLUTED_DIR = f'{BASE}/data/image_polluted'
os.makedirs(POLLUTED_DIR, exist_ok=True)

if BASE not in sys.path:
    sys.path.insert(0, BASE)

%pip install -q timm imagehash opencv-python-headless"""))

NB02.append(code("""# ============================================================
# 0-2. 사전등록 (DATASETS, POLLUTION_LEVELS)
# ============================================================
DATASETS = {
    'CIFAR10': {'loader': 'CIFAR10', 'n_classes': 10, 'image_size': 32, 'channels': 3},
    'FashionMNIST': {'loader': 'FashionMNIST', 'n_classes': 10, 'image_size': 28, 'channels': 1},
    'Flowers102': {'loader': 'Flowers102', 'n_classes': 102, 'image_size': 224, 'channels': 3},
}
POLLUTION_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
RANDOM_SEED = 42
SAMPLE_CAP = 5000  # 폴루션 후 DSC 계산 시 sample_cap

def load_train(ds_name):
    if ds_name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(f'{DATA_DIR}/CIFAR10', train=True, download=True)
    if ds_name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(f'{DATA_DIR}/FashionMNIST', train=True, download=True)
    if ds_name == 'Flowers102':
        return torchvision.datasets.Flowers102(f'{DATA_DIR}/Flowers102', split='train', download=True)

print(f'데이터셋: {list(DATASETS.keys())}, 강도: {POLLUTION_LEVELS}')"""))

NB02.append(md("""## 1. Polluter 5종 + DSC 점수"""))

NB02.append(code("""# ============================================================
# 1-1. polluter import + DSC import
# ============================================================
from dsc_framework import compute_dsc_image
from dsc_framework.image_polluters import (
    CompletenessImagePolluter, NoiseInjectionPolluter, BlurPolluter,
    ClassBalanceImagePolluter, LabelSwapPolluter,
)

def create_polluters(level, seed=RANDOM_SEED):
    return [
        ('completeness_image', CompletenessImagePolluter(level=level, random_seed=seed)),
        ('noise_injection', NoiseInjectionPolluter(level=level, random_seed=seed)),
        ('blur', BlurPolluter(level=level, random_seed=seed)),
        ('class_balance', ClassBalanceImagePolluter(level=level, random_seed=seed)),
        ('label_swap', LabelSwapPolluter(level=level, random_seed=seed)),
    ]
print('Polluter 5종 정의 완료')"""))

NB02.append(code("""# ============================================================
# 1-2. dataset → numpy 변환 + 폴루션 적용 + DSC
# ============================================================
def dataset_to_arrays(ds, sample_cap=None, random_state=1):
    images, labels = [], []
    n = len(ds) if sample_cap is None else min(len(ds), sample_cap)
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(ds))[:n] if sample_cap else range(n)
    for i in idx:
        img, lbl = ds[i]
        images.append(np.array(img))
        labels.append(int(lbl))
    return images, labels


from time import time

dsc_rows = []
total_start = time()

for ds_name in DATASETS:
    print(f'\\n=== {ds_name} ===')
    train_ds = load_train(ds_name)
    images_clean, labels_clean = dataset_to_arrays(train_ds, sample_cap=SAMPLE_CAP, random_state=1)
    print(f'  loaded {len(images_clean)} images')

    # baseline DSC
    res_base = compute_dsc_image(images_clean, labels_clean, sample_cap=SAMPLE_CAP)
    print(f'  baseline DSC = {res_base[\"score\"]} ({res_base[\"grade\"]})')
    dsc_rows.append({'dataset': ds_name, 'polluter': 'none', 'level': 0.0, **res_base})

    # 폴루션
    for level in POLLUTION_LEVELS:
        for polluter_name, polluter in create_polluters(level):
            t0 = time()
            try:
                pi, pl = polluter.pollute(images_clean, labels_clean)
                res_p = compute_dsc_image(pi, pl, sample_cap=SAMPLE_CAP)
                # 폴루션 데이터를 디스크에 저장 (03 노트북이 다시 사용)
                pol_dir = f'{POLLUTED_DIR}/{ds_name}/{polluter_name}_{int(level*100)}'
                os.makedirs(pol_dir, exist_ok=True)
                np.savez_compressed(f'{pol_dir}/data.npz',
                                    images=np.array([np.asarray(img) for img in pi], dtype=object),
                                    labels=np.array(pl))
                dsc_rows.append({'dataset': ds_name, 'polluter': polluter_name, 'level': level, **res_p})
                elapsed = time() - t0
                print(f'  {polluter_name:<22s} L={level:.2f}  DSC={res_p[\"score\"]:6.2f}  Δ={res_p[\"score\"]-res_base[\"score\"]:+.2f}  ({elapsed:.0f}s)')
            except Exception as e:
                print(f'  {polluter_name:<22s} L={level:.2f}  ERROR: {e}')

print(f'\\n총 {len(dsc_rows)}건 ({time() - total_start:.0f}초)')"""))

NB02.append(code("""# ============================================================
# 2-3. 결과 저장
# ============================================================
df_dsc = pd.DataFrame(dsc_rows)
out_path = f'{RESULTS_DIR}/dsc_scores_image.csv'
df_dsc.to_csv(out_path, index=False)
print(f'DSC 점수 저장: {out_path} (총 {len(df_dsc)}건)')
print('--- 노트북 02 이미지 cell 완료 ---')
df_dsc.head(15)"""))

make_nb(NB02, '02_pollution_and_dsc_image.ipynb')


# =========================================================================
# 노트북 03 — Training (Image)
# =========================================================================

NB03 = []
NB03.append(md("""# 03. Training & Evaluation (Image Cell)

**Phase 2**: 폴루션 데이터셋 × 모델 5개 학습 → accuracy 측정.

체크포인트: (dataset, polluter, level, model) 단위로 skip.
GPU 시간 변수 — Colab Pro+ 권장.

---"""))

NB03.append(code("""# ============================================================
# 0-1. 환경 + 의존성
# ============================================================
from google.colab import drive; drive.mount('/content/drive')
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader
from time import time

BASE = '/content/drive/MyDrive/capstone/dsc'
RESULTS_DIR = f'{BASE}/results'
DATA_DIR = f'{BASE}/data/image'
POLLUTED_DIR = f'{BASE}/data/image_polluted'

if BASE not in sys.path: sys.path.insert(0, BASE)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
%pip install -q timm"""))

NB03.append(code("""# ============================================================
# 0-2. 사전등록 + 모델 정의
# ============================================================
DATASETS = {
    'CIFAR10': {'n_classes': 10, 'image_size': 32, 'channels': 3},
    'FashionMNIST': {'n_classes': 10, 'image_size': 28, 'channels': 1},
    'Flowers102': {'n_classes': 102, 'image_size': 224, 'channels': 3},
}
MODEL_NAMES = ['ResNet18', 'EfficientNetB0', 'MobileNetV3small', 'ViTTiny', 'CNNSimple']
EPOCHS = 30  # 정식 (사전등록 ADR-014)
BATCH_SIZE = 128
LR = 1e-3


def get_model(model_name, n_classes, in_channels=3):
    if model_name == 'ResNet18':
        m = tvm.resnet18(weights=None)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    if model_name == 'EfficientNetB0':
        m = tvm.efficientnet_b0(weights=None)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        return m
    if model_name == 'MobileNetV3small':
        m = tvm.mobilenet_v3_small(weights=None)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        return m
    if model_name == 'ViTTiny':
        import timm
        return timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                 num_classes=n_classes, in_chans=in_channels)
    if model_name == 'CNNSimple':
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

print('모델 정의 완료')"""))

NB03.append(code("""# ============================================================
# 0-3. Dataset wrapper (numpy → tensor with transform)
# ============================================================
class NumpyDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        from PIL import Image
        img = self.images[i]
        if isinstance(img, np.ndarray):
            arr = img.squeeze() if img.ndim == 3 and img.shape[-1] == 1 else img
            img = Image.fromarray(arr) if arr.ndim == 2 else Image.fromarray(arr[..., :3])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])


def get_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_eval(model, train_ds, test_ds, epochs=EPOCHS):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    # eval
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
    return correct / total

print('학습 함수 정의 완료')"""))

NB03.append(md("""## 1. 실험 목록 + 체크포인트"""))

NB03.append(code("""# ============================================================
# 1-1. 실험 목록 스캔 (POLLUTED_DIR + clean baseline)
# ============================================================
def load_train_images(ds_name):
    \"\"\"clean train (sample_cap 적용된 numpy arrays).\"\"\"
    raw_path = f'{DATA_DIR}/{ds_name}'
    if ds_name == 'CIFAR10':
        ds = torchvision.datasets.CIFAR10(raw_path, train=True, download=True)
    elif ds_name == 'FashionMNIST':
        ds = torchvision.datasets.FashionMNIST(raw_path, train=True, download=True)
    elif ds_name == 'Flowers102':
        ds = torchvision.datasets.Flowers102(raw_path, split='train', download=True)
    images, labels = [], []
    rng = np.random.RandomState(1)
    idx = rng.permutation(len(ds))[:5000]
    for i in idx:
        img, lbl = ds[i]
        images.append(np.array(img)); labels.append(int(lbl))
    return images, labels


def load_test_images(ds_name):
    raw_path = f'{DATA_DIR}/{ds_name}'
    if ds_name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(raw_path, train=False, download=True)
    if ds_name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(raw_path, train=False, download=True)
    if ds_name == 'Flowers102':
        return torchvision.datasets.Flowers102(raw_path, split='test', download=True)


def load_polluted(ds_name, polluter, level):
    npz = np.load(f'{POLLUTED_DIR}/{ds_name}/{polluter}_{int(level*100)}/data.npz', allow_pickle=True)
    return list(npz['images']), npz['labels'].tolist()


# 실험 목록
experiments = []
for ds_name in DATASETS:
    # baseline
    experiments.append({'dataset': ds_name, 'polluter': 'none', 'level': 0.0})
    # 폴루션
    pol_dir = f'{POLLUTED_DIR}/{ds_name}'
    if os.path.isdir(pol_dir):
        for folder in sorted(os.listdir(pol_dir)):
            if not os.path.isfile(f'{pol_dir}/{folder}/data.npz'):
                continue
            parts = folder.rsplit('_', 1)
            experiments.append({'dataset': ds_name, 'polluter': parts[0], 'level': int(parts[1])/100})

print(f'실험 목록: {len(experiments)}건 × 모델 {len(MODEL_NAMES)}개 = {len(experiments)*len(MODEL_NAMES)}회 학습')"""))

NB03.append(code("""# ============================================================
# 2-1. 학습 루프 (체크포인트 지원)
# ============================================================
perf_path = f'{RESULTS_DIR}/model_performance_image.csv'

if os.path.isfile(perf_path):
    df_perf = pd.read_csv(perf_path)
    existing_keys = set(df_perf.apply(lambda r: f\"{r['dataset']}|{r['polluter']}|{r['level']}|{r['model']}\", axis=1))
    perf_rows = df_perf.to_dict('records')
    print(f'기존 결과 {len(perf_rows)}건 로드')
else:
    existing_keys, perf_rows = set(), []

total_start = time()
completed = skipped = 0
errors = []

for i, exp in enumerate(experiments):
    ds_name = exp['dataset']; meta = DATASETS[ds_name]
    # train images
    if exp['polluter'] == 'none':
        try: train_images, train_labels = load_train_images(ds_name)
        except Exception as e: print(f'load fail {ds_name}: {e}'); continue
    else:
        try: train_images, train_labels = load_polluted(ds_name, exp['polluter'], exp['level'])
        except Exception as e: print(f'load fail {ds_name}/{exp[\"polluter\"]}_{int(exp[\"level\"]*100)}: {e}'); continue

    # test (clean)
    test_raw = load_test_images(ds_name)
    test_images, test_labels = [], []
    for img, lbl in test_raw:
        test_images.append(np.array(img)); test_labels.append(int(lbl))

    transform = get_transform(image_size=224)
    train_ds = NumpyDataset(train_images, train_labels, transform)
    test_ds = NumpyDataset(test_images, test_labels, transform)

    for model_name in MODEL_NAMES:
        key = f\"{ds_name}|{exp['polluter']}|{exp['level']}|{model_name}\"
        if key in existing_keys:
            skipped += 1; continue
        try:
            t0 = time()
            model = get_model(model_name, meta['n_classes'])
            acc = train_eval(model, train_ds, test_ds, epochs=EPOCHS)
            elapsed = time() - t0
            row = {'dataset': ds_name, 'polluter': exp['polluter'], 'level': exp['level'],
                   'model': model_name, 'accuracy': round(acc, 4), 'epochs': EPOCHS}
            perf_rows.append(row); existing_keys.add(key); completed += 1
            print(f'  [{i+1}/{len(experiments)}] {ds_name}/{exp[\"polluter\"]}_{int(exp[\"level\"]*100)}/{model_name} acc={acc:.4f} ({elapsed:.0f}s)')
        except Exception as e:
            errors.append({'key': key, 'error': str(e)})
            print(f'  [{i+1}] {ds_name}/{model_name} ERROR: {e}')
        # 중간 저장
        if (completed + skipped) % 5 == 0:
            pd.DataFrame(perf_rows).to_csv(perf_path, index=False)

pd.DataFrame(perf_rows).to_csv(perf_path, index=False)
print(f'\\n학습 완료: 완료={completed}, 스킵={skipped}, 에러={len(errors)} ({time()-total_start:.0f}s)')"""))

make_nb(NB03, '03_training_image.ipynb')


# =========================================================================
# 노트북 04 — Scoreboard (Image)
# =========================================================================

NB04 = []
NB04.append(md("""# 04. Scoreboard (Image Cell)

회귀 cell의 04 노트북과 동일 분석 (DSC vs accuracy):
- 산점도, 라인, 히트맵, 박스플롯
- Pearson r, Spearman ρ, 비선형 RF 5-fold R²
- Polluter hold-out, 모델별 r
- 검증 기준: r ≥ 0.4

---"""))

NB04.append(code("""# ============================================================
# 0. 환경 + 데이터 로드
# ============================================================
from google.colab import drive; drive.mount('/content/drive')
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

BASE = '/content/drive/MyDrive/capstone/dsc'
RESULTS_DIR = f'{BASE}/results'
CHARTS_DIR = f'{RESULTS_DIR}/charts_image'
os.makedirs(CHARTS_DIR, exist_ok=True)
if BASE not in sys.path: sys.path.insert(0, BASE)

dsc = pd.read_csv(f'{RESULTS_DIR}/dsc_scores_image.csv')
perf = pd.read_csv(f'{RESULTS_DIR}/model_performance_image.csv')
print(f'DSC: {len(dsc)}, perf: {len(perf)}')

merged = perf.merge(dsc[['dataset','polluter','level','score','grade']],
                    on=['dataset','polluter','level'])
merged = merged.rename(columns={'score': 'dsc_score'})
print(f'merged: {len(merged)}')"""))

NB04.append(code("""# ============================================================
# 1. 산점도 + 라인 + 박스플롯
# ============================================================
plt.figure(figsize=(10, 6))
for m, sub in merged.groupby('model'):
    plt.scatter(sub['dsc_score'], sub['accuracy'], label=m, alpha=0.6, s=30)
plt.xlabel('DSC Score (image cell)'); plt.ylabel('Accuracy')
plt.title('DSC ↔ accuracy 산점도 (image cell)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(f'{CHARTS_DIR}/01_scatter.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=merged, x='grade', y='accuracy', order=['A','B','C','D'], palette='RdYlGn_r')
plt.title('DSC 등급별 accuracy (image cell)')
plt.savefig(f'{CHARTS_DIR}/02_grade_box.png', dpi=150)
plt.show()"""))

NB04.append(code("""# ============================================================
# 2. 통계 검증 + 가설 판정
# ============================================================
x = merged['dsc_score'].values; y = merged['accuracy'].values
r_p, p_p = pearsonr(x, y); r_s, p_s = spearmanr(x, y)
print(f'Pearson r = {r_p:+.4f}, Spearman ρ = {r_s:+.4f}')

# 비선형
X = merged[['dsc_score']].values; rf_folds = []
for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    rf_folds.append(r2_score(y[te], rf.predict(X[te])))
print(f'비선형 RF 5-fold R² = {np.mean(rf_folds):.4f} ± {np.std(rf_folds):.4f}')

# 모델별
print('\\n모델별 r:')
for m, sub in merged.groupby('model'):
    rr, pp = pearsonr(sub['dsc_score'], sub['accuracy'])
    print(f'  {m:<22s} r={rr:+.4f} p={pp:.2e} n={len(sub)}')

# polluter hold-out
print('\\nPolluter hold-out:')
hold_pass = 0; n_pol = 0
for hp in sorted(merged['polluter'].unique()):
    if hp == 'none': continue
    n_pol += 1
    sub = merged[merged.polluter != hp]
    rr, _ = pearsonr(sub['dsc_score'], sub['accuracy'])
    pass_ = rr >= 0.4; hold_pass += int(pass_)
    print(f'  {hp:<22s} r={rr:+.4f}  {\"PASS\" if pass_ else \"FAIL\"}')

# 가설 판정
verdict = {
    'H1 r ≥ 0.4': r_p >= 0.4,
    'H2 ρ ≥ 0.4': r_s >= 0.4,
    'H3 비선형 우위': np.mean(rf_folds) > r_p**2,
    'H4 모든 모델 양의 r': all(pearsonr(s['dsc_score'], s['accuracy'])[0] > 0
                              for _, s in merged.groupby('model')),
    'H5 polluter hold-out ≥4/5': hold_pass >= 4,
}
print('\\n=== 가설 판정 ===')
for k, v in verdict.items():
    print(f'  {\"✅\" if v else \"❌\"} {k}')
n_pass = sum(verdict.values())
print(f'\\n종합: {n_pass}/5 PASS')
if n_pass == 5:
    print('🎉 image cell Phase 2 통과')"""))

make_nb(NB04, '04_scoreboard_image.ipynb')

print('\\n=== 이미지 cell 노트북 4개 생성 완료 ===')
