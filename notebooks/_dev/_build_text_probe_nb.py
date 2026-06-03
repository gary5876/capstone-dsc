"""03_training_text_probe.ipynb 생성기 + 사전검증.

- 검증된 로직 문자열(LOGIC)을 구문검사 + _load_train_test disjoint(mock) 검사 후
  노트북으로 조립. 기존 파일은 건드리지 않음(신규 .ipynb 만 씀).
"""
import io
import json
import os
import tempfile
import py_compile

LOGIC = r'''# ============================================================
# 1. 덮어쓰기 방지 가드 + 출력 경로 (★ 새 파일에만 기록)
# ============================================================
OUT = f'{RESULTS}/text_probe_metrics.csv'        # 유일한 쓰기 대상 (신규)
DSC_SWEEP = f'{RESULTS}/text_dsc_sweep.csv'       # 읽기 전용 입력
PROTECTED = {'text_train_metrics.csv', 'text_train_metrics_dev.csv', 'text_dsc_sweep.csv',
             'text_baseline_dsc.csv', 'dsc_scores.csv', 'model_performance.csv', 'merged_results.csv'}
assert os.path.basename(OUT) not in PROTECTED, f'OUT={OUT} 가 보호 파일과 충돌 - 중단'

# ============================================================
# 2. 설정 (30분 하드캡 권장값; 숫자만 바꿔 조절)
# ============================================================
TRAIN_N = int(os.environ.get('TEXT_TRAIN_N', 800))
TEST_N  = int(os.environ.get('TEXT_TEST_N', 250))
BATCH   = int(os.environ.get('TEXT_BATCH', 64))
SEED    = 42
LEVELS  = [0.1, 0.25, 0.5, 0.75, 0.9]   # text_dsc_sweep.csv 와 동일한 비-baseline 단계

# ============================================================
# 3. imports - 기존 모듈 재사용 (DSC 임베딩 / probe / polluter)
# ============================================================
from dsc_framework.text_cell import _extract_features
from dsc_framework.perf_probe import evaluate_probes
from dsc_framework.text_polluters import (
    CompletenessTextPolluter, NoiseInjectionTextPolluter, WordShufflePolluter,
    ClassBalanceTextPolluter, LabelSwapTextPolluter,
    TargetDistributionSkewTextPolluter, TargetNoiseTextPolluter,
)

# polluter 이름은 text_dsc_sweep.csv 의 polluter 컬럼과 정확히 일치 (merge 키)
POLLUTERS_CLS = {
    'completeness_text':    CompletenessTextPolluter,
    'noise_injection_text': NoiseInjectionTextPolluter,
    'word_shuffle':         WordShufflePolluter,
    'class_balance':        ClassBalanceTextPolluter,
    'label_swap':           LabelSwapTextPolluter,
}
POLLUTERS_REG = {
    'completeness_text':        CompletenessTextPolluter,
    'noise_injection_text':     NoiseInjectionTextPolluter,
    'word_shuffle':             WordShufflePolluter,
    'target_distribution_skew': TargetDistributionSkewTextPolluter,
    'target_noise':             TargetNoiseTextPolluter,
}

DATASET_SPECS = [
    ('ag_news',   'fancyzhx/ag_news',               'classification'),
    ('imdb',      'stanfordnlp/imdb',               'classification'),
    ('20news',    'SetFit/20_newsgroups',           'classification'),
    ('yelp_full', 'Yelp/yelp_review_full',          'regression'),
    ('amazon_en', 'SetFit/amazon_reviews_multi_en', 'regression'),
    ('sst5',      'SetFit/sst5',                    'regression'),
]

# ============================================================
# 4. helpers - train/test 분할은 항상 disjoint (누수 방지)
# ============================================================
def _to_lists(ds_split, task):
    texts = list(ds_split['text'])
    labels = list(ds_split['label'])
    labels = [float(y) for y in labels] if task == 'regression' else [int(y) for y in labels]
    return texts, labels

def _subsample(ds_split, n, task, seed=42):
    if n is not None and len(ds_split) > n:
        ds_split = ds_split.shuffle(seed=seed).select(range(n))
    return _to_lists(ds_split, task)

def _load_train_test(ds, task):
    """test split 있으면 사용(서로 다른 split -> 자동 disjoint).
    없으면 train 을 셔플해 앞쪽 TEST_N / 그 다음 TRAIN_N 으로 잘라 disjoint 보장."""
    if 'test' in ds:
        tr_t, tr_y = _subsample(ds['train'], TRAIN_N, task, seed=SEED)
        te_t, te_y = _subsample(ds['test'],  TEST_N,  task, seed=SEED)
        return tr_t, tr_y, te_t, te_y
    full = ds['train'].shuffle(seed=SEED)
    n_te = min(TEST_N, max(1, len(full) // 5))
    te_t, te_y = _to_lists(full.select(range(n_te)), task)
    upper = min(n_te + TRAIN_N, len(full))
    tr_t, tr_y = _to_lists(full.select(range(n_te, upper)), task)
    return tr_t, tr_y, te_t, te_y

# ============================================================
# 5. resume - 기존 OUT(신규 파일)에서 끝난 config skip
# ============================================================
done = set()
if os.path.isfile(OUT):
    _prev = pd.read_csv(OUT)
    done = set(zip(_prev['dataset'], _prev['polluter'], _prev['level'].astype(float)))
    print(f'[resume] 기존 {OUT} 에서 {len(done)} config 완료 - skip')

def _append_rows(rows):
    df = pd.DataFrame(rows)
    header = not os.path.isfile(OUT)
    df.to_csv(OUT, mode='w' if header else 'a', header=header, index=False)

# ============================================================
# 6. sweep (config 단위 try/except - 하나 실패해도 전체 진행)
# ============================================================
def main():
    from datasets import load_dataset
    t0 = time.time(); n_done = n_skip = n_err = 0
    for ds_name, hf_id, task in DATASET_SPECS:
        print(f'\n=== {ds_name} ({task}) - load {hf_id} ===', flush=True)
        try:
            ds = load_dataset(hf_id)
            tr_texts, tr_y, te_texts, te_y = _load_train_test(ds, task)
        except Exception as e:
            print(f'    [SKIP dataset] {ds_name} 로드 실패: {e}', flush=True); continue
        print(f'    train={len(tr_texts)} test={len(te_texts)}', flush=True)
        Xte, _ = _extract_features(te_texts, sample_cap=None, batch_size=BATCH)
        yte = np.asarray(te_y, dtype=float if task == 'regression' else int)

        polluters = POLLUTERS_CLS if task == 'classification' else POLLUTERS_REG
        configs = [('none', 0.0)] + [(p, lv) for p in polluters for lv in LEVELS]
        for pname, lv in configs:
            key = (ds_name, pname, float(lv))
            if key in done:
                n_skip += 1; continue
            try:
                if pname == 'none':
                    ptr_texts, ptr_y = list(tr_texts), list(tr_y)
                else:
                    ptr_texts, ptr_y = polluters[pname](level=lv, random_seed=SEED).pollute(
                        list(tr_texts), list(tr_y))
                if len(ptr_texts) < 5:
                    print(f'    [skip] {pname}_{lv}: 오염 후 표본부족({len(ptr_texts)})'); continue
                Xtr, _ = _extract_features(ptr_texts, sample_cap=None, batch_size=BATCH)
                ytr = np.asarray(ptr_y, dtype=float if task == 'regression' else int)
                scores = evaluate_probes(Xtr, ytr, Xte, yte, task)
                rows = [{'dataset': ds_name, 'polluter': pname, 'level': float(lv),
                         'method': 'probe', 'model': k, 'score': v, 'task': task}
                        for k, v in scores.items() if not k.startswith('_') and v is not None]
                if rows:
                    _append_rows(rows); done.add(key); n_done += 1
                    _mean = np.mean([r['score'] for r in rows])
                    print(f'    {pname:24s} lv={lv:<4} probe평균={_mean:+.3f} '
                          f'({n_done} done, {time.time()-t0:.0f}s)', flush=True)
            except Exception as e:
                n_err += 1
                print(f'    [ERR] {ds_name}/{pname}_{lv}: {type(e).__name__}: {e}', flush=True)
    print(f'\n[완료] 신규 {n_done}, skip {n_skip}, 에러 {n_err}. {time.time()-t0:.0f}s -> {OUT}')
    _quick_scoreboard()

def _quick_scoreboard():
    """text_dsc_sweep.csv(읽기 전용)와 merge -> 데이터셋별 r 출력. 파일 안 씀."""
    if not (os.path.isfile(OUT) and os.path.isfile(DSC_SWEEP)):
        print('(scoreboard 생략: 파일 없음)'); return
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        print('(scipy 없음 - scoreboard 생략)'); return
    perf = pd.read_csv(OUT)
    dsc = pd.read_csv(DSC_SWEEP)
    pm = perf.groupby(['dataset', 'polluter', 'level'])['score'].mean().reset_index(name='probe_mean')
    score_col = 'dsc_score' if 'dsc_score' in dsc.columns else ('score' if 'score' in dsc.columns else None)
    if score_col is None:
        print('(text_dsc_sweep.csv 점수 컬럼 없음 - 생략)'); return
    dsc_s = dsc[['dataset', 'polluter', 'level', score_col]].drop_duplicates(['dataset', 'polluter', 'level'])
    m = pm.merge(dsc_s, on=['dataset', 'polluter', 'level'])
    print('\n=== 데이터셋별 r (DSC <-> probe 평균) ===')
    for ds_name, sub in m.groupby('dataset'):
        if len(sub) < 4:
            print(f'  {ds_name:12s} n={len(sub):2d} (부족)'); continue
        rp, _ = pearsonr(sub[score_col], sub['probe_mean'])
        rs, _ = spearmanr(sub[score_col], sub['probe_mean'])
        print(f'  {ds_name:12s} n={len(sub):2d}  pearson={rp:+.3f}  spearman={rs:+.3f}  [{"PASS" if rp>=0.40 else "below"}]')
'''

SETUP = '''# ============================================================
# 0. Colab 셋업 + BASE 탐색 (Drive 마운트)
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
%pip -q install datasets transformers

import os, sys, glob, time
import numpy as np
import pandas as pd

def _find_base():
    env = os.environ.get('DSC_BASE')
    if env and os.path.isfile(f'{env}/dsc_framework/__init__.py'):
        return env
    root = '/content/drive/MyDrive'
    for c in [f'{root}/capstone/dsc', f'{root}/dsc', f'{root}/capstone-dsc']:
        if os.path.isfile(f'{c}/dsc_framework/__init__.py'):
            return c
    for pat in [f'{root}/*/dsc_framework/__init__.py', f'{root}/*/*/dsc_framework/__init__.py']:
        for hit in glob.glob(pat):
            return os.path.dirname(os.path.dirname(hit))
    raise RuntimeError('dsc_framework/ 못 찾음 - Drive 마운트/동기화 확인')

BASE = _find_base()
RESULTS = f'{BASE}/results'
if BASE not in sys.path:
    sys.path.insert(0, BASE)
print('BASE =', BASE)
'''

MD = '''# 텍스트 cell - probe 기반 FULL sweep (분류 3 + 회귀 3)

finetune(무거움->dev만 가능) 대신 **이미지 cell과 동일한 frozen DistilBERT 임베딩 + probe**(ADR-019 방식)로
성능을 재서, 전 6데이터셋 x 전 오염단계를 실제로 완주한다. -> "텍스트 full sweep"을 둔갑 아닌 실제 실행으로 달성.

- 분류(ag_news/imdb/20news): F1(macro) / 회귀(yelp/amazon/sst5): R2(0 clip)
- 오염은 train만, 평가는 clean test / train·test 항상 disjoint(누수 방지)

**덮어쓰기 방지**: 결과는 신규 파일 `results/text_probe_metrics.csv` 에만 기록. 기존 CSV는 읽기 전용. per-config 체크포인트+resume.

실행: 위에서부터 셀 순서대로. 30분 권장값 TRAIN_N=800 / TEST_N=250 (셀 안 숫자만 조절).
'''


def _validate():
    # 검증 1: 구문
    tf = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8')
    tf.write("import os, time\nimport numpy as np\nimport pandas as pd\nRESULTS='/tmp'\n"
             + LOGIC + "\nmain\n_quick_scoreboard\n")
    tf.close()
    py_compile.compile(tf.name, doraise=True)
    print('검증1 OK: 구문 통과')

    # 검증 2: _load_train_test disjoint (mock) — dsc 루트를 path에 추가해야 import 됨
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    os.environ['TEXT_TRAIN_N'] = '30'; os.environ['TEXT_TEST_N'] = '10'
    ns = {'os': os, 'time': __import__('time'), 'np': __import__('numpy'),
          'pd': __import__('pandas'), 'RESULTS': '/tmp'}
    exec(LOGIC, ns)

    class FakeSplit:
        def __init__(self, t, l): self.t = t; self.l = l
        def __len__(self): return len(self.t)
        def __getitem__(self, k): return self.t if k == 'text' else self.l
        def shuffle(self, seed=0):
            import random
            idx = list(range(len(self.t))); random.Random(seed).shuffle(idx)
            return FakeSplit([self.t[i] for i in idx], [self.l[i] for i in idx])
        def select(self, rng):
            r = list(rng); return FakeSplit([self.t[i] for i in r], [self.l[i] for i in r])

    ds_a = {'train': FakeSplit([f'tr{i}' for i in range(200)], [i % 4 for i in range(200)]),
            'test':  FakeSplit([f'te{i}' for i in range(100)], [i % 4 for i in range(100)])}
    trt, _, tet, _ = ns['_load_train_test'](ds_a, 'classification')
    assert set(trt).isdisjoint(set(tet)), 'branch-a 누수!'
    print(f'검증2a OK: test split 존재 -> train {len(trt)}/test {len(tet)} disjoint')

    ds_b = {'train': FakeSplit([f'r{i}' for i in range(100)], [float(i % 5) for i in range(100)])}
    trt, _, tet, _ = ns['_load_train_test'](ds_b, 'regression')
    assert set(trt).isdisjoint(set(tet)) and len(trt) > 0 and len(tet) > 0, 'branch-b 누수!'
    print(f'검증2b OK: test 없음 -> train {len(trt)}/test {len(tet)} disjoint (누수 0)')
    os.environ.pop('TEXT_TRAIN_N'); os.environ.pop('TEXT_TEST_N')


def _build():
    def cc(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": s.splitlines(keepends=True)}

    def mc(s):
        return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

    nb = {"cells": [mc(MD), cc(SETUP), cc(LOGIC),
                    cc("# === 실행: 전 6데이터셋 probe sweep (결과는 text_probe_metrics.csv 에만) ===\nmain()\n"),
                    cc("# === scoreboard만 다시 보기 ===\n_quick_scoreboard()\n")],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"}},
          "nbformat": 4, "nbformat_minor": 5}
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, '..', '03_training_text_probe.ipynb')
    json.dump(nb, io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    back = json.load(open(out, encoding='utf-8'))
    print(f'검증3 OK: 노트북 생성 - cells={len(back["cells"])}, nbformat={back["nbformat"]} -> {os.path.normpath(out)}')


if __name__ == '__main__':
    _validate()
    _build()
