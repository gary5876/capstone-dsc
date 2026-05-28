"""LLM-based weight generator for DSC v5 framework (ADR-015 사전등록 구현).

운영 흐름:
1. (data_type, task, dataset_metadata) 입력
2. LLM 호출 -> 가중치 dict 출력
3. JSON schema validation: 메트릭 이름 일치, w in [0.01, 0.60], sum=1 (tol 0.01)
4. 검증 실패 시 fallback (cell별 default profile, ADR-009/011/014)

held-out 측정 직전 freeze 대상: prompt template version + LLM model + temperature.
freeze 항목 상세는 `documents/plans/20260511-01-합격선-heldout-사전등록.md` 참조.

사용 예시:
    from dsc_framework.llm_weight_generator import AnthropicWeightGenerator
    gen = AnthropicWeightGenerator(data_type='tabular', task='classification')
    result = gen.generate(dataset_metadata={'schema': ..., 'target_summary': ...})
    weights = result.weights  # fallback 사용 여부는 result.used_fallback 확인

CLAUDE.md "외부 요청 사전 검증 필수" 준수 사항:
- 본 모듈은 anthropic SDK 사용. `pip install anthropic` + 환경변수 ANTHROPIC_API_KEY 필요.
- 첫 호출 시 튜닝 set 1 dataset로 sanity check 권장.
- held-out 측정 직전 prompt + model + temperature freeze 후 측정 (plan 20260511-01 §3).
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classification_cell import DEFAULT_WEIGHTS_CLASSIFICATION
from .image_cell import DEFAULT_WEIGHTS_IMAGE
from .regression_cell import DEFAULT_WEIGHTS_REGRESSION
from .text_cell import DEFAULT_WEIGHTS_TEXT
from .text_cell_regression import DEFAULT_WEIGHTS_TEXT_REG

PROMPT_DIR = Path(__file__).parent / "prompts"

DEFAULT_WEIGHT_PROFILES: dict[tuple[str, str], dict[str, float]] = {
    ('tabular', 'classification'): DEFAULT_WEIGHTS_CLASSIFICATION,
    ('tabular', 'regression'): DEFAULT_WEIGHTS_REGRESSION,
    ('image', 'classification'): DEFAULT_WEIGHTS_IMAGE,
    ('text', 'classification'): DEFAULT_WEIGHTS_TEXT,
    ('text', 'regression'): DEFAULT_WEIGHTS_TEXT_REG,
}

WEIGHT_BOUNDS: tuple[float, float] = (0.01, 0.60)
SUM_TOLERANCE: float = 0.01


@dataclass
class WeightGenerationResult:
    weights: dict[str, float]
    used_fallback: bool
    fallback_reason: str = ""
    raw_response: str = ""
    model: str = ""
    temperature: float = 0.0
    prompt_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            'weights': self.weights,
            'used_fallback': self.used_fallback,
            'fallback_reason': self.fallback_reason,
            'model': self.model,
            'temperature': self.temperature,
            'prompt_version': self.prompt_version,
            'raw_response': self.raw_response,
        }


class WeightGenerator(ABC):
    """Provider-agnostic LLM weight generator. 구현체는 `_call_llm`만 override."""

    def __init__(
        self,
        data_type: str,
        task: str,
        prompt_version: str = 'v1',
        temperature: float = 0.0,
    ):
        self.data_type = data_type
        self.task = task
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.cell_key = (data_type, task)
        if self.cell_key not in DEFAULT_WEIGHT_PROFILES:
            raise ValueError(
                f"Unsupported cell {self.cell_key}. "
                f"Supported: {sorted(DEFAULT_WEIGHT_PROFILES.keys())}")
        self.metric_names: list[str] = list(
            DEFAULT_WEIGHT_PROFILES[self.cell_key].keys())
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        path = PROMPT_DIR / f"weight_generator_{self.prompt_version}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {path}. "
                f"`dsc_framework/prompts/` 디렉토리에 weight_generator_{self.prompt_version}.txt 추가 필요.")
        return path.read_text(encoding='utf-8')

    def _build_user_message(self, dataset_metadata: dict[str, Any]) -> str:
        return self.prompt_template.format(
            data_type=self.data_type,
            task=self.task,
            metric_names_json=json.dumps(self.metric_names, ensure_ascii=False),
            dataset_metadata_json=json.dumps(
                dataset_metadata, ensure_ascii=False, indent=2),
            w_min=WEIGHT_BOUNDS[0],
            w_max=WEIGHT_BOUNDS[1],
        )

    @abstractmethod
    def _call_llm(self, user_message: str) -> str:
        """Provider-specific LLM call. Returns raw response string."""

    def generate(self, dataset_metadata: dict[str, Any]) -> WeightGenerationResult:
        user_message = self._build_user_message(dataset_metadata)
        try:
            raw_response = self._call_llm(user_message)
        except Exception as exc:  # noqa: BLE001 — fallback 위해 광범위 capture
            return self._fallback(f"llm_call_failed: {exc}", "")

        try:
            weights = self._parse_response(raw_response)
        except ValueError as exc:
            return self._fallback(f"parse_failed: {exc}", raw_response)

        is_valid, reason = self._validate_weights(weights)
        if not is_valid:
            return self._fallback(f"validation_failed: {reason}", raw_response)

        return WeightGenerationResult(
            weights=weights,
            used_fallback=False,
            raw_response=raw_response,
            model=getattr(self, 'model', ''),
            temperature=self.temperature,
            prompt_version=self.prompt_version,
        )

    def _parse_response(self, raw_response: str) -> dict[str, float]:
        text = raw_response.strip()
        if '```' in text:
            for chunk in text.split('```'):
                stripped = chunk.strip()
                if stripped.startswith('json'):
                    stripped = stripped[4:].strip()
                if stripped.startswith('{'):
                    text = stripped
                    break
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"non-JSON response: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("top-level JSON must be an object")
        weights = data.get('weights', data)
        if not isinstance(weights, dict):
            raise ValueError("'weights' must be an object")
        try:
            return {str(k): float(v) for k, v in weights.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weight values must be numeric: {exc}") from exc

    def _validate_weights(self, weights: dict[str, float]) -> tuple[bool, str]:
        expected = set(self.metric_names)
        actual = set(weights.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            return False, f"metric mismatch (missing={missing}, extra={extra})"
        w_min, w_max = WEIGHT_BOUNDS
        for name, w in weights.items():
            if not (w_min <= w <= w_max):
                return False, f"{name}={w:.4f} outside [{w_min},{w_max}]"
        total = sum(weights.values())
        if abs(total - 1.0) > SUM_TOLERANCE:
            return False, f"sum={total:.4f} not in [1-{SUM_TOLERANCE}, 1+{SUM_TOLERANCE}]"
        return True, ""

    def _fallback(self, reason: str, raw_response: str) -> WeightGenerationResult:
        return WeightGenerationResult(
            weights=dict(DEFAULT_WEIGHT_PROFILES[self.cell_key]),
            used_fallback=True,
            fallback_reason=reason,
            raw_response=raw_response,
            model=getattr(self, 'model', ''),
            temperature=self.temperature,
            prompt_version=self.prompt_version,
        )


class AnthropicWeightGenerator(WeightGenerator):
    """Anthropic Claude API 기반 weight generator.

    실행 전 확인:
    - `pip install anthropic`
    - 환경변수 ANTHROPIC_API_KEY 설정
    - held-out 측정 직전 model + temperature freeze (plan 20260511-01 §3-2)
    """

    def __init__(
        self,
        data_type: str,
        task: str,
        model: str = 'claude-sonnet-4-6',
        prompt_version: str = 'v1',
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(data_type, task, prompt_version, temperature)
        self.model = model
        self.max_tokens = max_tokens
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic  # noqa: PLC0415 — lazy import
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK 미설치. `pip install anthropic` 후 재시도.") from exc
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise RuntimeError(
                "환경변수 ANTHROPIC_API_KEY 미설정. .env 또는 secrets 매니저에서 로드 필요.")
        self._client = anthropic.Anthropic(api_key=api_key)

    def _call_llm(self, user_message: str) -> str:
        self._ensure_client()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{'role': 'user', 'content': user_message}],
        )
        text_parts = [
            getattr(block, 'text', '')
            for block in response.content
            if getattr(block, 'type', '') == 'text'
        ]
        return '\n'.join(text_parts).strip()


__all__ = [
    'AnthropicWeightGenerator',
    'WeightGenerator',
    'WeightGenerationResult',
    'DEFAULT_WEIGHT_PROFILES',
    'WEIGHT_BOUNDS',
    'SUM_TOLERANCE',
]
