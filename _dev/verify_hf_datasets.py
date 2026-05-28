"""HF datasets 6종 실제 load 검증.

페이지 GET이 아니라 datasets.load_dataset()을 실제 호출.
small split만 가져와서 schema(text 키, label 키, 행수)만 확인. mteb/amazon
같은 dataset script 차단 케이스가 페이지 GET으로 안 잡혀서 추가.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# 분류 3 + 회귀 3 = 6종. small split으로만.
TARGETS = [
    ('fancyzhx/ag_news',              None,  'test',          'text',  'label',           'classification'),
    ('stanfordnlp/imdb',              None,  'test[:200]',    'text',  'label',           'classification'),
    ('SetFit/20_newsgroups',          None,  'test[:200]',    'text',  'label',           'classification'),
    ('Yelp/yelp_review_full',         None,  'test[:200]',    'text',  'label',           'regression'),
    ('SetFit/amazon_reviews_multi_en', None,  'test[:200]',   'text',  'label',           'regression'),
    ('SetFit/sst5',                   None,  'test[:200]',    'text',  'label',           'regression'),
]


def main():
    from datasets import load_dataset
    print('=' * 70)
    print('HF DATASETS LOAD VERIFICATION')
    print('=' * 70)
    results = []
    for repo, config, split, text_key, label_key, task in TARGETS:
        try:
            if config:
                ds = load_dataset(repo, config, split=split)
            else:
                ds = load_dataset(repo, split=split)
            assert text_key in ds.column_names, f"missing {text_key}: {ds.column_names}"
            assert label_key in ds.column_names, f"missing {label_key}: {ds.column_names}"
            sample = ds[0]
            t_preview = (sample[text_key][:60] + '...') if len(sample[text_key]) > 60 else sample[text_key]
            print(f"OK  {repo:40s} n={len(ds):5d} label={sample[label_key]} text={t_preview!r}")
            results.append((repo, True, None))
        except Exception as e:
            tb = traceback.format_exc().splitlines()[-3:]
            print(f"FAIL {repo:40s} → {type(e).__name__}: {e}")
            print('     ' + '\n     '.join(tb))
            results.append((repo, False, str(e)))

    print()
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f'PASS {n_ok}/{len(results)}')
    if n_ok < len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
