#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VAC-MQR: version- and API-constrained query rewriting for SE RAG.

The online selector is label-free. Ground-truth relevance labels are used only
by evaluate(), preventing test-label leakage into candidate selection.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class QAExample:
    query: str
    relevant_doc_ids: List[str]
    answer: str = ""
    repair_doc_ids: List[str] = field(default_factory=list)
    group: str = "clean"
    pair_id: str = ""


@dataclass
class TriggerCalibrationExample:
    query: str
    needs_rewrite: bool


@dataclass
class AnswerEvaluationRecord:
    """Human- or judge-verified answer scores; this module only aggregates them."""
    correctness: float
    faithfulness: float
    answer_relevance: float
    citation_precision: float
    citation_recall: float
    fix_success: float | None = None


@dataclass
class SEConstraints:
    projects: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    versions: List[str] = field(default_factory=list)
    apis: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    intent: str = "information"

    def protected(self) -> List[str]:
        values = (self.projects + self.languages + self.frameworks + self.versions
                  + self.apis + self.errors + self.files)
        return list(dict.fromkeys(x for x in values if x))


@dataclass
class Candidate:
    text: str
    view: str
    constraints: SEConstraints


@dataclass
class VACMQRResult:
    query: str
    rewrite_triggered: bool
    intent: str
    uncertainty: float
    constraints: SEConstraints
    candidates: List[str]
    filter_details: List[Dict]
    retrieval_details: List[Dict]
    strategy: str
    final_queries: List[str]
    trigger_prediction_set: List[int] = field(default_factory=list)


class SplitConformalTrigger:
    """Binary split-conformal predictor for rewrite decisions.

    Coverage is marginal and requires exchangeability between calibration and
    test examples. Label 1 means that rewriting is needed.
    """
    def __init__(self, alpha: float = .10):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.quantile: float | None = None

    def fit(self, probabilities: Sequence[float], labels: Sequence[int]) -> None:
        if len(probabilities) != len(labels) or not labels:
            raise ValueError("non-empty probabilities and labels must have equal length")
        scores = []
        for probability, label in zip(probabilities, labels):
            p_rewrite = float(np.clip(probability, 0, 1))
            p_true = p_rewrite if label == 1 else 1-p_rewrite
            scores.append(1-p_true)
        ordered = np.sort(np.asarray(scores))
        rank = min(math.ceil((len(ordered)+1)*(1-self.alpha)), len(ordered))-1
        self.quantile = float(ordered[max(rank, 0)])

    def predict(self, probability: float) -> List[int]:
        if self.quantile is None:
            raise RuntimeError("conformal trigger has not been calibrated")
        p_rewrite = float(np.clip(probability, 0, 1))
        probabilities = {0: 1-p_rewrite, 1: p_rewrite}
        return [label for label, value in probabilities.items()
                if 1-value <= self.quantile]


class LinearCandidateRanker:
    """Small ridge ranker trained only on labelled training queries."""
    feature_names = ("mean_similarity", "precision_proxy", "symbol_coverage",
                     "version_match")

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self.weights: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.weights is not None

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("ranker requires a non-empty 2-D feature matrix and targets")
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        z = (x-self.mean)/self.std
        design = np.column_stack([np.ones(len(z)), z])
        penalty = self.ridge*np.eye(design.shape[1]); penalty[0, 0] = 0
        self.weights = np.linalg.solve(design.T@design+penalty, design.T@y)

    def score(self, features: Sequence[float]) -> float:
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("candidate ranker has not been trained")
        z = (np.asarray(features, dtype=float)-self.mean)/self.std
        return float(np.r_[1.0, z]@self.weights)


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder(Embedder):
    """Character n-gram embedder for smoke tests, not final experiments."""
    def __init__(self, dim: int = 768, ngram: int = 3):
        self.dim, self.ngram = dim, ngram

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            blocks = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_.:#/-]+", text.lower())
            tokens = [b[i:i+n] for b in blocks for n in range(1, self.ngram + 1)
                      for i in range(max(0, len(b) - n + 1))] or [text]
            for token in tokens:
                value = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                matrix[row, (value >> 1) % self.dim] += 1 if value & 1 else -1
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceBERTEmbedder(Embedder):
    def __init__(self, model: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> List[Document]: ...


class EmbeddingRetriever(Retriever):
    def __init__(self, documents: List[Document], embedder: Embedder):
        self.documents, self.embedder = documents, embedder
        self.vectors = embedder.embed([d.text for d in documents]) if documents else None

    def search(self, query: str, k: int) -> List[Document]:
        if self.vectors is None or k <= 0:
            return []
        similarities = self.vectors @ self.embedder.embed([query])[0]
        order = np.argsort(-similarities)[:k]
        return [Document(self.documents[i].id, self.documents[i].text,
                         self.documents[i].metadata, float(similarities[i])) for i in order]


INTENT_TERMS = {
    "bug_fix": ["修复", "解决", "怎么办", "如何处理", "workaround", "fix", "resolved issue"],
    "bug_diagnosis": ["报错", "异常", "失败", "error", "exception", "原因", "为什么", "root cause"],
    "api_usage": ["怎么调用", "如何使用", "参数", "返回值", "示例", "用法"],
    "dependency": ["依赖", "版本", "兼容", "安装", "冲突"],
    "configuration": ["配置", "设置", "环境变量", "yaml", "json"],
    "code_explanation": ["解释", "作用", "做什么", "原理"],
    "performance": ["性能", "速度", "内存", "延迟", "优化"],
    "comparison": ["区别", "对比", "比较", "哪个好"],
}

LANGUAGES = ["Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "Kotlin"]
FRAMEWORKS = ["Spring Boot", "Spring", "Django", "Flask", "FastAPI", "React", "Vue", "Angular",
              "PyTorch", "TensorFlow", "pandas", "NumPy", "Redis", "Docker", "Kubernetes"]


class SEQueryAnalyzer:
    def normalize(self, query: str) -> str:
        for old, new in [("咋办", "如何解决"), ("怎么弄", "如何处理"),
                         ("为啥", "为什么"), ("跑不起来", "运行失败")]:
            query = query.replace(old, new)
        return query.strip()

    def detect_intent(self, query: str) -> str:
        normalized = self.normalize(query).lower()
        scores = {name: sum(term.lower() in normalized for term in terms)
                  for name, terms in INTENT_TERMS.items()}
        return max(scores, key=scores.get) if max(scores.values(), default=0) else "information"

    def classification_entropy(self, query: str) -> float:
        normalized = self.normalize(query).lower()
        counts = np.array([sum(term.lower() in normalized for term in terms)
                           for terms in INTENT_TERMS.values()], dtype=float)
        probabilities = np.exp(counts - counts.max())
        probabilities /= probabilities.sum()
        return float(-(probabilities * np.log(probabilities + 1e-12)).sum()
                     / math.log(len(probabilities)))

    def extract_constraints(self, query: str) -> SEConstraints:
        version_mentions = re.findall(
            r"(?i)(?:(?:python|java|node(?:\.js)?|spring(?: boot)?|django|react|vue|"
            r"pandas|numpy|docker)\s*v?\d+(?:\.\d+){0,2}(?:[-._][a-z0-9]+)?|"
            r"v\d+(?:\.\d+){0,2}(?:[-._][a-z0-9]+)?|"
            r"\b\d+\.\d+(?:\.\d+)?(?:[-._][a-z0-9]+)?\b)", query)
        versions = []
        for value in version_mentions:
            match = re.search(r"(?i)\d+(?:\.\d+){0,2}(?:[-._][a-z0-9]+)?$", value)
            if match:
                versions.append(match.group(0))
        errors = re.findall(
            r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Failure)\b|"
            r"\b(?:HTTP\s*)?[1-5]\d{2}\b|\b[A-Z]{2,}[-_]\d+\b", query)
        backticks = re.findall(r"`([^`]+)`", query)
        dotted = re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:\(\))?", query)
        calls = re.findall(r"\b[A-Za-z_]\w*\(\)", query)
        code_symbols = re.findall(r"\b(?:[A-Z][a-z0-9]+){2,}\b", query)
        error_set = {x.lower() for x in errors}
        apis = list(dict.fromkeys(x for x in backticks + dotted + calls + code_symbols
                                  if x.lower() not in error_set))
        files = re.findall(r"(?i)\b[\w.-]+\.(?:py|java|js|ts|tsx|jsx|go|rs|cpp|h|"
                           r"xml|yaml|yml|json|toml|ini|gradle|properties)\b", query)
        languages = [x for x in LANGUAGES
                     if re.search(rf"(?i)(?<!\w){re.escape(x)}(?!\w)", query)]
        frameworks = [x for x in FRAMEWORKS if x.lower() in query.lower()]
        frameworks = [x for x in frameworks
                      if not any(x.lower() != y.lower() and x.lower() in y.lower()
                                 for y in frameworks)]
        issue_ids = re.findall(r"(?i)(?:issue\s*)?#\d+", query)
        projects = list(dict.fromkeys(frameworks + issue_ids))
        return SEConstraints(projects, languages, frameworks,
                             list(dict.fromkeys(versions)), apis,
                             list(dict.fromkeys(errors)), list(dict.fromkeys(files)),
                             self.detect_intent(query))

    @staticmethod
    def protect(query: str, cons: SEConstraints) -> Tuple[str, Dict[str, str]]:
        """Replace extracted software entities with immutable placeholders."""
        protected_query, mapping = query, {}
        categories = [("VER", cons.versions), ("API", cons.apis),
                      ("ERR", cons.errors), ("FILE", cons.files),
                      ("FRAME", cons.frameworks), ("LANG", cons.languages)]
        occupied = set()
        for category, values in categories:
            index = 0
            for value in sorted(values, key=len, reverse=True):
                if not value or value.lower() in occupied:
                    continue
                pattern = re.compile(re.escape(value), re.IGNORECASE)
                if pattern.search(protected_query):
                    index += 1
                    placeholder = f"<{category}_{index}>"
                    protected_query = pattern.sub(placeholder, protected_query)
                    mapping[placeholder] = value
                    occupied.add(value.lower())
        return protected_query, mapping

    @staticmethod
    def restore(text: str, mapping: Dict[str, str]) -> str:
        for placeholder, value in mapping.items():
            text = text.replace(placeholder, value)
        return text

    def generate_views(self, query: str, cons: SEConstraints) -> Dict[str, List[str]]:
        protected_query, mapping = self.protect(query, cons)
        normalized = self.restore(self.normalize(protected_query), mapping)
        protected = " ".join(cons.protected())
        expansion = {
            "bug_fix": ["fix", "workaround", "resolved issue", "pull request"],
            "bug_diagnosis": ["root cause", "stack trace", "issue"],
            "api_usage": ["API documentation", "usage example", "parameters"],
            "dependency": ["compatibility", "release notes", "migration guide"],
            "configuration": ["configuration documentation", "example"],
            "code_explanation": ["implementation", "documentation"],
            "performance": ["benchmark", "profiling", "optimization"],
        }.get(cons.intent, ["documentation", "issue"])
        keyword = " ".join([protected] + expansion).strip()
        semantic = f"Technical documentation or resolved issue for: {normalized}"
        decomposed: List[str] = []
        if cons.intent in {"bug_fix", "bug_diagnosis"}:
            decomposed = [f"{protected} root cause".strip(),
                          f"{protected} verified fix compatibility".strip()]
        # Safe pseudo-document: no invented version, API, cause, or solution.
        pseudo = (f"A software engineering document about {protected or normalized}; "
                  f"it preserves the exact API, version and error and discusses {cons.intent}.")
        return {"conservative": [normalized], "keyword": [keyword],
                "semantic": [semantic], "decompose": decomposed, "pseudo_doc": [pseudo]}


class ConstraintVerifier:
    """Deterministically verifies all entities recognized in the source query."""
    def __init__(self, analyzer: SEQueryAnalyzer):
        self.analyzer = analyzer

    @staticmethod
    def _canonical(value: str) -> str:
        return re.sub(r"\(\)$", "", value.strip()).lower()

    def verify(self, original: SEConstraints, rewritten_text: str) -> Dict[str, float | bool]:
        rewritten = self.analyzer.extract_constraints(rewritten_text)

        def retention(expected: List[str], actual: List[str]) -> float:
            if not expected:
                return 1.0
            actual_set = {self._canonical(value) for value in actual}
            return sum(self._canonical(value) in actual_set for value in expected)/len(expected)

        scores = {"version": retention(original.versions, rewritten.versions),
                  "api": retention(original.apis, rewritten.apis),
                  "error": retention(original.errors, rewritten.errors),
                  "file": retention(original.files, rewritten.files)}
        scores["verified"] = all(value == 1.0 for value in scores.values())
        return scores


@dataclass
class RewriteQuality:
    accepted: bool
    reasons: List[str]
    constraint_preservation: float
    negation_preserved: bool


class CandidateQualityValidator:
    """Deterministic gate applied before retrieval labels or learned ranking."""

    _unk = re.compile(r"(?i)(?:<unk>|\bunk\b)")
    _negation = re.compile(
        r"(?i)(?<!\w)(?:not|no|never|without|cannot|can't|dont|don't|doesnt|"
        r"doesn't|isnt|isn't|wont|won't|shouldnt|shouldn't)(?!\w)")

    def __init__(self, analyzer: SEQueryAnalyzer | None = None):
        self.analyzer = analyzer or SEQueryAnalyzer()
        self.verifier = ConstraintVerifier(self.analyzer)

    @staticmethod
    def canonical(text: str) -> str:
        return " ".join(text.casefold().split())

    def validate(self, original: str, candidate: str,
                 seen: set[str] | None = None) -> RewriteQuality:
        reasons: List[str] = []
        normalized = self.canonical(candidate)
        if not normalized:
            reasons.append("empty")
        if self._unk.search(candidate):
            reasons.append("literal_unk")
        if normalized == self.canonical(original):
            reasons.append("identical_to_original")
        if seen is not None and normalized in seen:
            reasons.append("duplicate_candidate")
        original_negative = bool(self._negation.search(original))
        candidate_negative = bool(self._negation.search(candidate))
        negation_preserved = original_negative == candidate_negative
        if not negation_preserved:
            reasons.append("negation_flip")

        constraints = self.analyzer.extract_constraints(original)
        verification = self.verifier.verify(constraints, candidate)
        retained = [float(verification[name]) for name in ("version", "api", "error", "file")]
        for name in ("frameworks", "languages"):
            expected = getattr(constraints, name)
            if expected:
                lowered = candidate.casefold()
                retained.append(sum(value.casefold() in lowered for value in expected) / len(expected))
        preservation = float(np.mean(retained)) if retained else 1.0
        if not bool(verification["verified"]):
            reasons.append("constraint_loss")
        if constraints.frameworks and not all(x.casefold() in candidate.casefold()
                                              for x in constraints.frameworks):
            reasons.append("framework_loss")
        if constraints.languages and not all(x.casefold() in candidate.casefold()
                                             for x in constraints.languages):
            reasons.append("language_loss")
        if seen is not None:
            seen.add(normalized)
        return RewriteQuality(not reasons, list(dict.fromkeys(reasons)),
                              preservation, negation_preserved)


UTILITY_FEATURES = (
    "information_gain", "query_candidate_cosine", "token_jaccard", "length_ratio",
    "original_top1", "candidate_top1", "top1_delta", "original_margin",
    "candidate_margin", "top10_mean_delta", "top10_overlap",
    "original_top10_mean", "original_top10_std", "original_top10_range",
    "original_top10_entropy", "candidate_top10_mean", "candidate_top10_std",
    "candidate_top10_range", "candidate_top10_entropy", "top10_std_delta",
    "top10_entropy_delta", "constraint_preservation", "candidate_rank",
)

ENHANCED_RETRIEVAL_FEATURES = UTILITY_FEATURES + (
    "top100_overlap", "top100_rrf_agreement", "top10_rank_stability",
    "score_distribution_correlation", "original_top100_mean", "original_top100_std",
    "candidate_top100_mean", "candidate_top100_std", "top100_mean_delta",
    "top100_std_delta", "top1_document_stable",
)

INTERACTION_FEATURES = ENHANCED_RETRIEVAL_FEATURES + (
    "interaction_score", "interaction_margin_to_original",
)

UTILITY_GRADES = (-1, 0, 1, 2)
LCB_EPSILON = 1e-12


def passes_gain_lcb(lower_bound: float, threshold: float) -> bool:
    """Apply the strict LCB gate consistently despite floating-point noise."""
    return float(lower_bound) > float(threshold) + LCB_EPSILON


class UtilityRanker:
    """Delta-RR ranker with a four-level utility head and one-sided LCB."""

    def __init__(self, feature_names: Sequence[str] = UTILITY_FEATURES,
                 ridge: float = 1e-2):
        self.feature_names = tuple(feature_names)
        self.ridge = float(ridge)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.gain_weights: np.ndarray | None = None
        self.positive_weights: np.ndarray | None = None
        self.conditional_gain_weights: np.ndarray | None = None
        self.lambdarank_model: str | None = None
        self.lambdarank_gain_calibration: np.ndarray | None = None
        self._lambdarank_booster: Any | None = None
        self.cqr_lower_weights: np.ndarray | None = None
        self.positive_stumps: List[Dict[str, float]] = []
        self.conditional_gain_stumps: List[Dict[str, float]] = []
        self.cqr_stumps: List[Dict[str, float]] = []
        self.cqr_correction = 0.0
        self.cqr_quantile = .10
        self.harm_weights: np.ndarray | None = None
        self.grade_weights: np.ndarray | None = None
        self.gain_residual_quantile = 0.0
        self.gain_scale_weights: np.ndarray | None = None
        self.gain_scale_floor = 1e-4
        self.gain_lcb_scale_quantile = 1.0
        self.lcb_alpha = .10
        self.gain_stumps: List[Dict[str, float]] = []
        self.harm_stumps: List[Dict[str, float]] = []
        self.boost_learning_rate = .05

    @staticmethod
    def _fit_stump(x: np.ndarray, residual: np.ndarray,
                   min_leaf: int = 20) -> Dict[str, float] | None:
        best: Dict[str, float] | None = None
        best_error = float("inf")
        for feature in range(x.shape[1]):
            for threshold in np.unique(np.quantile(x[:, feature], np.linspace(.1, .9, 9))):
                left = x[:, feature] <= threshold
                if left.sum() < min_leaf or (~left).sum() < min_leaf:
                    continue
                left_value = float(residual[left].mean())
                right_value = float(residual[~left].mean())
                prediction = np.where(left, left_value, right_value)
                error = float(np.mean(np.square(residual - prediction)))
                if error < best_error:
                    best_error = error
                    best = {"feature": float(feature), "threshold": float(threshold),
                            "left": left_value, "right": right_value}
        return best

    @staticmethod
    def _apply_stumps(x: np.ndarray, stumps: Sequence[Dict[str, float]],
                      learning_rate: float) -> np.ndarray:
        output = np.zeros(len(x), dtype=np.float64)
        for stump in stumps:
            feature = int(stump["feature"])
            output += learning_rate * np.where(
                x[:, feature] <= stump["threshold"], stump["left"], stump["right"])
        return output

    @property
    def fitted(self) -> bool:
        return (self.gain_weights is not None
                and (self.grade_weights is not None or self.harm_weights is not None))

    def _matrix(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.asarray([[float(row[name]) for name in self.feature_names] for row in rows],
                          dtype=np.float64)

    def _fit_gain_scale(self, z: np.ndarray, residuals: np.ndarray) -> None:
        target = np.log(np.abs(residuals) + self.gain_scale_floor)
        penalty = self.ridge * np.eye(z.shape[1]); penalty[0, 0] = 0
        self.gain_scale_weights = np.linalg.solve(
            z.T @ z + penalty, z.T @ target)

    def _predict_gain_scale_from_z(self, z: np.ndarray) -> np.ndarray:
        if self.gain_scale_weights is None:
            return np.ones(len(z), dtype=np.float64)
        return np.maximum(np.exp(np.clip(z @ self.gain_scale_weights, -12, 4)),
                          self.gain_scale_floor)

    def fit(self, rows: Sequence[Dict[str, float]], delta_rr: Sequence[float],
            utility_grades: Sequence[int], *, logistic_steps: int = 800,
            learning_rate: float = .05, lcb_alpha: float = .10) -> None:
        x = self._matrix(rows)
        y = np.asarray(delta_rr, dtype=np.float64)
        grades = np.asarray(utility_grades, dtype=np.int64)
        if x.ndim != 2 or len(x) == 0 or len(x) != len(y) or len(y) != len(grades):
            raise ValueError("ranker inputs must be non-empty and aligned")
        if not set(grades.tolist()).issubset(UTILITY_GRADES):
            raise ValueError(f"utility grades must be in {UTILITY_GRADES}")
        if not 0 < lcb_alpha < 1:
            raise ValueError("lcb_alpha must be between 0 and 1")
        self.lcb_alpha = float(lcb_alpha)
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        penalty = self.ridge * np.eye(z.shape[1]); penalty[0, 0] = 0
        self.gain_weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        gain_prediction = z @ self.gain_weights
        self.gain_stumps = []
        for _ in range(80):
            stump = self._fit_stump(z[:, 1:], y - gain_prediction)
            if stump is None:
                break
            self.gain_stumps.append(stump)
            gain_prediction += self.boost_learning_rate * np.where(
                z[:, 1 + int(stump["feature"])] <= stump["threshold"],
                stump["left"], stump["right"])
        self.gain_residual_quantile = max(0.0, float(np.quantile(
            gain_prediction - y, 1.0 - self.lcb_alpha)))
        self._fit_gain_scale(z, y - gain_prediction)

        # Two-stage expected gain: P(delta>0) * E[delta | delta>0].
        positive = (y > 0.0).astype(np.float64)
        positive_weights = np.zeros(z.shape[1], dtype=np.float64)
        for _ in range(logistic_steps):
            logits = np.clip(z @ positive_weights, -30, 30)
            probability = 1.0 / (1.0 + np.exp(-logits))
            gradient = z.T @ (probability - positive) / len(z)
            gradient[1:] += self.ridge * positive_weights[1:] / len(z)
            positive_weights -= learning_rate * gradient
        self.positive_weights = positive_weights
        positive_raw = z @ positive_weights
        self.positive_stumps = []
        for _ in range(120):
            probability = 1.0 / (1.0 + np.exp(-np.clip(positive_raw, -30, 30)))
            stump = self._fit_stump(z[:, 1:], positive - probability)
            if stump is None:
                break
            self.positive_stumps.append(stump)
            positive_raw += self.boost_learning_rate * np.where(
                z[:, 1 + int(stump["feature"])] <= stump["threshold"],
                stump["left"], stump["right"])
        positive_rows = positive > 0.0
        conditional_penalty = self.ridge * np.eye(z.shape[1]); conditional_penalty[0, 0] = 0
        self.conditional_gain_weights = np.linalg.solve(
            z[positive_rows].T @ z[positive_rows] + conditional_penalty,
            z[positive_rows].T @ y[positive_rows])
        conditional_prediction = z[positive_rows] @ self.conditional_gain_weights
        self.conditional_gain_stumps = []
        for _ in range(120):
            stump = self._fit_stump(
                z[positive_rows, 1:], y[positive_rows] - conditional_prediction)
            if stump is None:
                break
            self.conditional_gain_stumps.append(stump)
            conditional_prediction += self.boost_learning_rate * np.where(
                z[positive_rows, 1 + int(stump["feature"])] <= stump["threshold"],
                stump["left"], stump["right"])

        # Lower conditional quantile head. A disjoint role conformalizes it later.
        lower_weights = np.zeros(z.shape[1], dtype=np.float64)
        for _ in range(1600):
            residual = y - z @ lower_weights
            gradient = -(z.T @ (self.cqr_quantile - (residual < 0.0))) / len(z)
            gradient[1:] += self.ridge * lower_weights[1:] / len(z)
            lower_weights -= .01 * gradient
        self.cqr_lower_weights = lower_weights
        lower_prediction = z @ lower_weights
        self.cqr_stumps = []
        for _ in range(200):
            residual = y - lower_prediction
            pseudo_residual = self.cqr_quantile - (residual < 0.0).astype(float)
            stump = self._fit_stump(z[:, 1:], pseudo_residual)
            if stump is None:
                break
            self.cqr_stumps.append(stump)
            lower_prediction += .01 * np.where(
                z[:, 1 + int(stump["feature"])] <= stump["threshold"],
                stump["left"], stump["right"])

        weights = np.zeros((len(UTILITY_GRADES), z.shape[1]), dtype=np.float64)
        targets = np.asarray([UTILITY_GRADES.index(int(value)) for value in grades])
        counts = np.bincount(targets, minlength=len(UTILITY_GRADES)).astype(float)
        class_weight = len(targets) / np.maximum(len(UTILITY_GRADES) * counts, 1.0)
        sample_weight = class_weight[targets]
        for _ in range(logistic_steps):
            logits = np.clip(z @ weights.T, -30, 30)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            residual = probabilities
            residual[np.arange(len(z)), targets] -= 1.0
            gradient = (residual * sample_weight[:, None]).T @ z / len(z)
            gradient[:, 1:] += self.ridge * weights[:, 1:] / len(z)
            weights -= learning_rate * gradient
        self.grade_weights = weights
        self.harm_weights = None
        raw_harm = z @ weights[0]
        harmful = (grades == -1).astype(np.float64)
        self.harm_stumps = []
        for _ in range(120):
            probability = 1.0 / (1.0 + np.exp(-np.clip(raw_harm, -30, 30)))
            stump = self._fit_stump(z[:, 1:], harmful - probability)
            if stump is None:
                break
            self.harm_stumps.append(stump)
            raw_harm += self.boost_learning_rate * np.where(
                z[:, 1 + int(stump["feature"])] <= stump["threshold"],
                stump["left"], stump["right"])

    def fit_groupwise(self, rows: Sequence[Dict[str, float]], delta_rr: Sequence[float],
                      utility_grades: Sequence[int], groups: Sequence[int], *,
                      gain_objective: str = "two_stage", **kwargs: Any) -> None:
        """Fit gain by within-query pairwise logistic ranking."""
        if gain_objective not in {"two_stage", "legacy_groupwise"}:
            raise ValueError(f"unsupported gain objective: {gain_objective}")
        self.fit(rows, delta_rr, utility_grades, **kwargs)
        x = self._matrix(rows); y = np.asarray(delta_rr, dtype=np.float64)
        group_values = np.asarray(groups)
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        pairs = [(left, right) for group in np.unique(group_values)
                 for left in np.flatnonzero(group_values == group)
                 for right in np.flatnonzero(group_values == group) if y[left] > y[right]]
        if not pairs:
            return
        differences = np.asarray([z[left] - z[right] for left, right in pairs])
        weights = np.zeros(z.shape[1], dtype=np.float64)
        for _ in range(700):
            margins = np.clip(differences @ weights, -30, 30)
            probabilities = 1.0 / (1.0 + np.exp(-margins))
            gradient = differences.T @ (probabilities - 1.0) / len(differences)
            gradient[1:] += self.ridge * weights[1:] / len(differences)
            weights -= .03 * gradient
        pointwise_prediction = z @ self.gain_weights + self._apply_stumps(
            z[:, 1:], self.gain_stumps, self.boost_learning_rate)
        rank_scores = z @ weights
        calibration = np.column_stack([np.ones(len(rank_scores)),
                                       pointwise_prediction, rank_scores])
        penalty = self.ridge * np.eye(3); penalty[0, 0] = 0
        blend = np.linalg.solve(calibration.T @ calibration + penalty,
                                calibration.T @ y)
        self.gain_weights = blend[1] * self.gain_weights + blend[2] * weights
        self.gain_weights[0] += blend[0]
        # Keep the gain head pairwise-consistent. Pointwise residual boosting can
        # undo within-query ordering, so it is intentionally disabled here.
        self.gain_stumps = []
        if gain_objective == "legacy_groupwise":
            self.positive_weights = None
            self.conditional_gain_weights = None
            self.positive_stumps = []
            self.conditional_gain_stumps = []
        expected_gain, _ = self.predict(rows)
        self._fit_gain_scale(z, y - expected_gain)

    def fit_lambdarank(self, rows: Sequence[Dict[str, float]], delta_rr: Sequence[float],
                       utility_grades: Sequence[int], groups: Sequence[int], *,
                       num_boost_round: int = 200, **kwargs: Any) -> None:
        """Fit a LightGBM LambdaRank gain head over query-level candidate groups."""
        import lightgbm as lgb

        self.fit(rows, delta_rr, utility_grades, **kwargs)
        x = self._matrix(rows)
        y = np.asarray(delta_rr, dtype=np.float64)
        grades = np.asarray(utility_grades, dtype=np.int64)
        group_values = np.asarray(groups)
        unique_groups = list(dict.fromkeys(group_values.tolist()))
        order = np.concatenate([np.flatnonzero(group_values == group)
                                for group in unique_groups])
        group_sizes = [int(np.count_nonzero(group_values == group))
                       for group in unique_groups]
        labels = grades + 1
        train_data = lgb.Dataset(x[order], label=labels[order], group=group_sizes,
                                 feature_name=list(self.feature_names), free_raw_data=False)
        booster = lgb.train({
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "label_gain": [0, 1, 2, 4],
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": self.ridge,
            "verbosity": -1,
            "seed": 101,
            "num_threads": 0,
        }, train_data, num_boost_round=num_boost_round)
        self.lambdarank_model = booster.model_to_string()
        self._lambdarank_booster = booster
        scores = np.asarray(booster.predict(x), dtype=np.float64)
        variance = float(np.var(scores))
        slope = (float(np.cov(scores, y, bias=True)[0, 1]) / variance
                 if variance > 1e-12 else 0.0)
        slope = max(slope, 1e-8)
        intercept = float(np.mean(y) - slope * np.mean(scores))
        self.lambdarank_gain_calibration = np.asarray([intercept, slope], dtype=np.float64)
        self.positive_weights = None
        self.conditional_gain_weights = None
        self.positive_stumps = []
        self.conditional_gain_stumps = []
        self.cqr_lower_weights = None
        self.cqr_stumps = []
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        gain, _ = self.predict(rows)
        self._fit_gain_scale(z, y - gain)

    def predict(self, rows: Sequence[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("utility ranker has not been fitted")
        x = self._matrix(rows)
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        if self.lambdarank_model is not None:
            import lightgbm as lgb

            if self._lambdarank_booster is None:
                self._lambdarank_booster = lgb.Booster(model_str=self.lambdarank_model)
            scores = np.asarray(self._lambdarank_booster.predict(x), dtype=np.float64)
            gain = (self.lambdarank_gain_calibration[0]
                    + self.lambdarank_gain_calibration[1] * scores)
        elif self.positive_weights is not None and self.conditional_gain_weights is not None:
            positive_raw = z @ self.positive_weights + self._apply_stumps(
                z[:, 1:], self.positive_stumps, self.boost_learning_rate)
            positive_probability = 1.0 / (1.0 + np.exp(-np.clip(positive_raw, -30, 30)))
            conditional_gain = np.maximum(
                z @ self.conditional_gain_weights + self._apply_stumps(
                    z[:, 1:], self.conditional_gain_stumps, self.boost_learning_rate), 0.0)
            gain = positive_probability * conditional_gain
        else:
            gain = z @ self.gain_weights + self._apply_stumps(
                z[:, 1:], self.gain_stumps, self.boost_learning_rate)
        if self.grade_weights is not None:
            logits = np.clip(z @ self.grade_weights.T, -30, 30)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            harm = probabilities[:, 0]
        else:
            harm_raw = z @ self.harm_weights + self._apply_stumps(
                z[:, 1:], self.harm_stumps, self.boost_learning_rate)
            harm = 1.0 / (1.0 + np.exp(-np.clip(harm_raw, -30, 30)))
        return gain, harm

    def predict_with_lcb(self, rows: Sequence[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gain, harm = self.predict(rows)
        x = self._matrix(rows)
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        if self.cqr_lower_weights is not None:
            lower_bound = np.minimum(
                z @ self.cqr_lower_weights + self._apply_stumps(
                    z[:, 1:], self.cqr_stumps, .01) - self.cqr_correction, gain)
        elif self.gain_scale_weights is None:
            lower_bound = gain - self.gain_residual_quantile
        else:
            lower_bound = gain - self.gain_lcb_scale_quantile * self._predict_gain_scale_from_z(z)
        return gain, lower_bound, harm

    def calibrate_cqr_lcb(self, rows: Sequence[Dict[str, float]],
                          delta_rr: Sequence[float], alpha: float = .10) -> None:
        """Conformalize the trained lower-quantile head on a disjoint role."""
        if self.cqr_lower_weights is None:
            raise RuntimeError("CQR lower-quantile head has not been fitted")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        targets = np.asarray(delta_rr, dtype=np.float64)
        x = self._matrix(rows)
        if not len(x) or len(x) != len(targets):
            raise ValueError("CQR calibration inputs must be non-empty and aligned")
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        lower_prediction = z @ self.cqr_lower_weights + self._apply_stumps(
            z[:, 1:], self.cqr_stumps, .01)
        scores = np.sort(lower_prediction - targets)
        rank = min(math.ceil((len(scores) + 1) * (1 - alpha)), len(scores)) - 1
        self.cqr_correction = max(0.0, float(scores[max(rank, 0)]))
        self.lcb_alpha = float(alpha)

    def calibrate_gain_lcb(self, rows: Sequence[Dict[str, float]],
                           delta_rr: Sequence[float], groups: Sequence[int] | None = None,
                           alpha: float = .10) -> None:
        """Calibrate a one-sided residual bound on selected held-out candidates."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        targets = np.asarray(delta_rr, dtype=np.float64)
        predictions, _ = self.predict(rows)
        if len(predictions) == 0 or len(predictions) != len(targets):
            raise ValueError("LCB calibration inputs must be non-empty and aligned")
        if groups is not None:
            group_values = np.asarray(groups)
            if len(group_values) != len(predictions):
                raise ValueError("LCB calibration groups must align with rows")
            selected = [indices[int(np.argmax(predictions[indices]))]
                        for group in np.unique(group_values)
                        for indices in [np.flatnonzero(group_values == group)]]
            predictions, targets = predictions[selected], targets[selected]
        x = self._matrix(rows)
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        scales = self._predict_gain_scale_from_z(z)
        if groups is not None:
            scales = scales[selected]
        normalized = np.sort((predictions - targets) / scales)
        rank = min(math.ceil((len(normalized) + 1) * (1 - alpha)), len(normalized)) - 1
        self.gain_lcb_scale_quantile = max(0.0, float(normalized[max(rank, 0)]))
        self.gain_residual_quantile = float(self.gain_lcb_scale_quantile * np.median(scales))
        self.lcb_alpha = float(alpha)

    def save(self, path: Path, metadata: Dict[str, Any] | None = None) -> None:
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("cannot save an unfitted utility ranker")
        payload = {"format": "vac-mqr-utility-ranker-v3", "feature_names": self.feature_names,
                   "ridge": self.ridge, "mean": self.mean.tolist(), "std": self.std.tolist(),
                   "gain_weights": self.gain_weights.tolist(),
                   "positive_weights": (self.positive_weights.tolist()
                                        if self.positive_weights is not None else None),
                   "conditional_gain_weights": (self.conditional_gain_weights.tolist()
                                                if self.conditional_gain_weights is not None else None),
                   "lambdarank_model": self.lambdarank_model,
                   "lambdarank_gain_calibration": (
                       self.lambdarank_gain_calibration.tolist()
                       if self.lambdarank_gain_calibration is not None else None),
                   "cqr_lower_weights": (self.cqr_lower_weights.tolist()
                                         if self.cqr_lower_weights is not None else None),
                   "cqr_correction": self.cqr_correction,
                   "cqr_quantile": self.cqr_quantile,
                   "positive_stumps": self.positive_stumps,
                   "conditional_gain_stumps": self.conditional_gain_stumps,
                   "cqr_stumps": self.cqr_stumps,
                   "harm_weights": self.harm_weights.tolist() if self.harm_weights is not None else None,
                   "grade_weights": self.grade_weights.tolist() if self.grade_weights is not None else None,
                   "utility_grades": UTILITY_GRADES,
                   "gain_residual_quantile": self.gain_residual_quantile,
                   "gain_scale_weights": (self.gain_scale_weights.tolist()
                                          if self.gain_scale_weights is not None else None),
                   "gain_scale_floor": self.gain_scale_floor,
                   "gain_lcb_scale_quantile": self.gain_lcb_scale_quantile,
                   "gain_lcb_method": "normalized one-sided residual conformal",
                   "lcb_alpha": self.lcb_alpha,
                   "boost_learning_rate": self.boost_learning_rate,
                   "gain_stumps": self.gain_stumps, "harm_stumps": self.harm_stumps,
                   "metadata": metadata or {}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "UtilityRanker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(payload["feature_names"], payload["ridge"])
        for name in ("mean", "std", "gain_weights"):
            setattr(model, name, np.asarray(payload[name], dtype=np.float64))
        for name in ("positive_weights", "conditional_gain_weights", "cqr_lower_weights"):
            if payload.get(name) is not None:
                setattr(model, name, np.asarray(payload[name], dtype=np.float64))
        model.lambdarank_model = payload.get("lambdarank_model")
        if payload.get("lambdarank_gain_calibration") is not None:
            model.lambdarank_gain_calibration = np.asarray(
                payload["lambdarank_gain_calibration"], dtype=np.float64)
        model.cqr_correction = float(payload.get("cqr_correction", 0.0))
        model.cqr_quantile = float(payload.get("cqr_quantile", .10))
        model.positive_stumps = list(payload.get("positive_stumps", []))
        model.conditional_gain_stumps = list(payload.get("conditional_gain_stumps", []))
        model.cqr_stumps = list(payload.get("cqr_stumps", []))
        if payload.get("grade_weights") is not None:
            model.grade_weights = np.asarray(payload["grade_weights"], dtype=np.float64)
        elif payload.get("harm_weights") is not None:
            model.harm_weights = np.asarray(payload["harm_weights"], dtype=np.float64)
        model.gain_residual_quantile = float(payload.get("gain_residual_quantile", 0.0))
        if payload.get("gain_scale_weights") is not None:
            model.gain_scale_weights = np.asarray(payload["gain_scale_weights"], dtype=np.float64)
        model.gain_scale_floor = float(payload.get("gain_scale_floor", 1e-4))
        model.gain_lcb_scale_quantile = float(payload.get("gain_lcb_scale_quantile", 1.0))
        model.lcb_alpha = float(payload.get("lcb_alpha", .10))
        model.boost_learning_rate = float(payload.get("boost_learning_rate", .05))
        model.gain_stumps = list(payload.get("gain_stumps", []))
        model.harm_stumps = list(payload.get("harm_stumps", []))
        return model


class HarmfulRiskModel:
    """Independent class-balanced harmful-rewrite probability model."""

    def __init__(self, feature_names: Sequence[str] = UTILITY_FEATURES,
                 ridge: float = 1e-2, learning_rate: float = .05):
        self.feature_names = tuple(feature_names)
        self.ridge = float(ridge)
        self.learning_rate = float(learning_rate)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.stumps: List[Dict[str, float]] = []

    @property
    def fitted(self) -> bool:
        return self.weights is not None

    def _matrix(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.asarray([[float(row[name]) for name in self.feature_names] for row in rows],
                          dtype=np.float64)

    def fit(self, rows: Sequence[Dict[str, float]], harmful: Sequence[int],
            *, logistic_steps: int = 1200, stump_count: int = 200) -> None:
        x = self._matrix(rows)
        labels = np.asarray(harmful, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or len(x) != len(labels):
            raise ValueError("harmful risk inputs must be non-empty and aligned")
        if not set(labels.tolist()).issubset({0.0, 1.0}) or labels.min() == labels.max():
            raise ValueError("harmful risk training requires both binary classes")
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        positive_weight = len(labels) / (2.0 * labels.sum())
        negative_weight = len(labels) / (2.0 * (len(labels) - labels.sum()))
        sample_weight = np.where(labels == 1, positive_weight, negative_weight)
        weights = np.zeros(z.shape[1], dtype=np.float64)
        for _ in range(logistic_steps):
            logits = np.clip(z @ weights, -30, 30)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = z.T @ ((probabilities - labels) * sample_weight) / len(z)
            gradient[1:] += self.ridge * weights[1:] / len(z)
            weights -= self.learning_rate * gradient
        self.weights = weights

        raw = z @ weights
        self.stumps = []
        for _ in range(stump_count):
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
            residual = (labels - probabilities) * sample_weight
            stump = UtilityRanker._fit_stump(z[:, 1:], residual)
            if stump is None:
                break
            self.stumps.append(stump)
            raw += self.learning_rate * np.where(
                z[:, 1 + int(stump["feature"])] <= stump["threshold"],
                stump["left"], stump["right"])

    def predict_proba(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("harmful risk model has not been fitted")
        x = self._matrix(rows)
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        raw = z @ self.weights + UtilityRanker._apply_stumps(
            z[:, 1:], self.stumps, self.learning_rate)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))

    def save(self, path: Path, metadata: Dict[str, Any] | None = None) -> None:
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("cannot save an unfitted harmful risk model")
        payload = {
            "format": "vac-mqr-harmful-risk-v1", "feature_names": self.feature_names,
            "ridge": self.ridge, "learning_rate": self.learning_rate,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "weights": self.weights.tolist(), "stumps": self.stumps,
            "metadata": metadata or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HarmfulRiskModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(payload["feature_names"], payload["ridge"], payload["learning_rate"])
        for name in ("mean", "std", "weights"):
            setattr(model, name, np.asarray(payload[name], dtype=np.float64))
        model.stumps = list(payload.get("stumps", []))
        return model


class ConformalRiskController:
    """Split-conformal harmful-label set plus a calibrated empirical risk gate.

    The conformal statement is marginal label coverage under exchangeability. It
    is not a conditional guarantee on the harmful rewrite rate of accepted rows.
    """

    def __init__(self, alpha: float = .10):
        self.predictor = SplitConformalTrigger(alpha)
        self.harm_probability_threshold = 1.0
        self.safe_probability_threshold = 0.0

    @property
    def quantile(self) -> float | None:
        return self.predictor.quantile

    def fit(self, harm_probabilities: Sequence[float], harmful_labels: Sequence[int]) -> None:
        self.predictor.fit(harm_probabilities, harmful_labels)
        if self.predictor.quantile is None:
            raise RuntimeError("conformal calibration failed")
        # Label 1 (harm) is excluded iff 1-p_harm > conformal quantile.
        self.safe_probability_threshold = max(0.0, 1.0 - self.predictor.quantile)

    def permits(self, harm_probability: float) -> bool:
        return (harm_probability < self.safe_probability_threshold
                and harm_probability <= self.harm_probability_threshold)

    def to_dict(self) -> Dict[str, float]:
        if self.quantile is None:
            raise RuntimeError("risk controller has not been calibrated")
        return {"alpha": self.predictor.alpha, "conformal_quantile": self.quantile,
                "conformal_safe_probability_threshold": self.safe_probability_threshold,
                "harm_probability_threshold": self.harm_probability_threshold}


def select_with_safe_arm(candidates: Sequence[Dict[str, Any]], ranker: UtilityRanker,
                         risk: ConformalRiskController, min_gain: float,
                         risk_model: HarmfulRiskModel | None = None) -> Dict[str, Any]:
    """Return a candidate only when both learned gates pass; otherwise original."""
    if not candidates:
        return {"selected": "original", "candidate_index": None,
                "predicted_gain": 0.0, "predicted_harm": 0.0}
    features = [row["features"] for row in candidates]
    gains, lower_bounds, legacy_harms = ranker.predict_with_lcb(features)
    harms = risk_model.predict_proba(features) if risk_model is not None else legacy_harms
    order = np.argsort(-gains)[:3]
    for index in order:
        if passes_gain_lcb(float(lower_bounds[index]), min_gain) and risk.permits(float(harms[index])):
            return {"selected": "rewrite", "candidate_index": int(index),
                    "predicted_gain": float(gains[index]),
                    "gain_lower_confidence_bound": float(lower_bounds[index]),
                    "predicted_harm": float(harms[index])}
    return {"selected": "original", "candidate_index": None,
            "predicted_gain": 0.0, "predicted_harm": 0.0}


class VACMQR:
    def __init__(self, analyzer: SEQueryAnalyzer, embedder: Embedder, retriever: Retriever,
                 *, trigger: float = .55, filter_threshold: float = .60,
                 top_k: int = 5, relevance_threshold: float = .30,
                 fuse_margin: float = .02, conformal_alpha: float = .10):
        self.analyzer, self.embedder, self.retriever = analyzer, embedder, retriever
        self.trigger, self.filter_threshold = trigger, filter_threshold
        self.top_k, self.relevance_threshold, self.fuse_margin = top_k, relevance_threshold, fuse_margin
        self.verifier = ConstraintVerifier(analyzer)
        self.quality_validator = CandidateQualityValidator(analyzer)
        self.conformal_trigger = SplitConformalTrigger(conformal_alpha)
        self.ranker = LinearCandidateRanker()
        self.utility_ranker: UtilityRanker | None = None
        self.risk_controller: ConformalRiskController | None = None
        self.min_predicted_gain = 0.0

    def configure_utility_selection(self, ranker: UtilityRanker,
                                    controller: ConformalRiskController,
                                    min_predicted_gain: float) -> None:
        self.utility_ranker = ranker
        self.risk_controller = controller
        self.min_predicted_gain = float(min_predicted_gain)

    def select_utility_candidates(self, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Select from externally featurized rewrites while keeping original permanently."""
        if self.utility_ranker is None or self.risk_controller is None:
            raise RuntimeError("utility selection is not configured")
        qualified = [row for row in candidates if row.get("quality_pass", False)]
        return select_with_safe_arm(qualified, self.utility_ranker, self.risk_controller,
                                    self.min_predicted_gain)

    def calibrate_trigger(self, examples: Sequence[TriggerCalibrationExample]) -> None:
        probabilities, labels = [], []
        for example in examples:
            cons = self.analyzer.extract_constraints(example.query)
            entropy = self.analyzer.classification_entropy(example.query)
            probabilities.append(self.uncertainty(example.query, entropy, cons))
            labels.append(int(example.needs_rewrite))
        self.conformal_trigger.fit(probabilities, labels)

    def uncertainty(self, query: str, entropy: float, cons: SEConstraints) -> float:
        documents = self.retriever.search(query, self.top_k)
        retrieval_u = 1.0 if not documents else float(np.clip(1-max(d.score for d in documents), 0, 1))
        surface_u = .20*bool(re.search(r"它|这个|那个|咋|怎么弄", query))
        surface_u += .15*bool(len(query) < 12)
        surface_u += .15*bool(cons.errors and not (cons.apis or cons.frameworks))
        surface_u += .10*bool("和" in query or "以及" in query)
        return round(float(np.clip(.40*entropy + .40*retrieval_u + surface_u, 0, 1)), 4)

    @staticmethod
    def canonical_symbol(value: str) -> str:
        return re.sub(r"\(\)$", "", value.strip()).lower()

    @classmethod
    def preservation(cls, expected: List[str], actual: List[str]) -> float:
        if not expected:
            return 1.0
        actual_values = {cls.canonical_symbol(value) for value in actual}
        return sum(cls.canonical_symbol(value) in actual_values for value in expected)/len(expected)

    @staticmethod
    def intent_compatible(original: str, rewritten: str) -> float:
        compatible = {
            "bug_fix": {"bug_fix", "bug_diagnosis"},
            "bug_diagnosis": {"bug_diagnosis", "bug_fix"},
        }
        return float(rewritten in compatible.get(original, {original}))

    def build_candidates(self, query: str, original: SEConstraints) -> Tuple[List[Candidate], List[Dict]]:
        candidates = [Candidate(query, "original", original)]
        for view, texts in self.analyzer.generate_views(query, original).items():
            candidates += [Candidate(text, view, self.analyzer.extract_constraints(text))
                           for text in texts if text and text != query]
        candidates = list({c.text: c for c in candidates}.values())
        vectors = self.embedder.embed([c.text for c in candidates])
        kept, details = [candidates[0]], []
        for candidate, vector in zip(candidates[1:], vectors[1:]):
            verification = self.verifier.verify(original, candidate.text)
            version = float(verification["version"])
            api = float(verification["api"])
            error = float(verification["error"])
            file_retention = float(verification["file"])
            framework = self.preservation(original.frameworks, candidate.constraints.frameworks)
            intent = self.intent_compatible(original.intent, candidate.constraints.intent)
            semantic = float(np.clip(vector @ vectors[0], 0, 1))
            hard_pass = bool(verification["verified"])
            score = (.18*version + .18*api + .14*error + .10*file_retention
                     + .08*framework + .14*intent + .18*semantic)
            accepted = hard_pass and score >= self.filter_threshold
            details.append({"candidate": candidate.text, "view": candidate.view,
                            "version": version, "api": api, "error": error,
                            "file": file_retention, "verified": hard_pass,
                            "framework": framework, "intent": intent,
                            "semantic": semantic, "score": score, "kept": accepted})
            if accepted:
                kept.append(candidate)
        return kept[:7], details

    def version_match(self, cons: SEConstraints, docs: List[Document]) -> float:
        if not cons.versions:
            return .5
        weights = np.asarray([1/math.log2(rank+1) for rank in range(1, len(docs)+1)])
        matches = []
        for document in docs:
            content = (document.text + " " + " ".join(document.metadata.values())).lower()
            matches.append(sum(version.lower() in content for version in cons.versions)
                           / len(cons.versions))
        return float(np.average(matches, weights=weights))

    @staticmethod
    def symbol_match(symbols: List[str], docs: List[Document]) -> float:
        if not symbols:
            return .5
        weights = np.asarray([1/math.log2(rank+1) for rank in range(1, len(docs)+1)])
        matches = []
        for document in docs:
            content = document.text.lower()
            matches.append(sum(symbol.lower() in content for symbol in symbols)/len(symbols))
        return float(np.average(matches, weights=weights))

    def select(self, candidates: List[Candidate]) -> Tuple[List[Dict], str, List[str]]:
        ranked = []
        for candidate in candidates:
            docs = self.retriever.search(candidate.text, self.top_k)
            if not docs:
                continue
            similarities = np.asarray([d.score for d in docs])
            mean_similarity = float(similarities.mean())
            precision_proxy = float((similarities >= self.relevance_threshold).mean())
            protected = candidate.constraints.apis + candidate.constraints.errors
            symbol_coverage = self.symbol_match(protected, docs)
            version_match = self.version_match(candidate.constraints, docs)
            features = [mean_similarity, precision_proxy, symbol_coverage, version_match]
            heuristic_proxy = (.35*mean_similarity + .20*precision_proxy
                               + .25*symbol_coverage + .20*version_match)
            proxy = self.ranker.score(features) if self.ranker.fitted else heuristic_proxy
            item = {"candidate": candidate.text, "view": candidate.view,
                    "proxy_score": proxy, "mean_similarity": mean_similarity,
                    "precision_proxy": precision_proxy, "symbol_coverage": symbol_coverage,
                    "version_match": version_match, "ranker": "learned" if self.ranker.fitted else "heuristic",
                    "doc_ids": [d.id for d in docs]}
            ranked.append((proxy, candidate.text, item))
        ranked.sort(reverse=True, key=lambda x: x[0])
        if not ranked:
            return [], "single", [candidates[0].text]
        strategy, final = "single", [ranked[0][1]]
        if len(ranked) > 1 and ranked[0][0]-ranked[1][0] < self.fuse_margin:
            strategy, final = "fused", [ranked[0][1], ranked[1][1]]
        return [x[2] for x in ranked], strategy, final

    def run(self, query: str) -> VACMQRResult:
        cons = self.analyzer.extract_constraints(query)
        uncertainty = self.uncertainty(query, self.analyzer.classification_entropy(query), cons)
        prediction_set: List[int] = []
        if self.conformal_trigger.quantile is not None:
            prediction_set = self.conformal_trigger.predict(uncertainty)
            # Rewrite only on a confident singleton decision. Ambiguous sets
            # retain the original query and avoid an unjustified intervention.
            should_rewrite = prediction_set == [1]
        else:
            should_rewrite = uncertainty >= self.trigger
        if not should_rewrite:
            return VACMQRResult(query, False, cons.intent, uncertainty, cons,
                                [query], [], [], "single", [query], prediction_set)
        candidates, filters = self.build_candidates(query, cons)
        retrieval, strategy, final = self.select(candidates)
        return VACMQRResult(query, True, cons.intent, uncertainty, cons,
                            [c.text for c in candidates], filters, retrieval, strategy, final,
                            prediction_set)

    def fit_ranker(self, examples: Sequence[QAExample]) -> None:
        """Fit on training queries only; never pass validation/test examples."""
        feature_rows, targets = [], []
        for example in examples:
            original = self.analyzer.extract_constraints(example.query)
            candidates, _ = self.build_candidates(example.query, original)
            relevant = set(example.relevant_doc_ids)
            for candidate in candidates:
                docs = self.retriever.search(candidate.text, self.top_k)
                if not docs:
                    continue
                similarities = np.asarray([d.score for d in docs])
                features = [float(similarities.mean()),
                            float((similarities >= self.relevance_threshold).mean()),
                            self.symbol_match(candidate.constraints.apis + candidate.constraints.errors, docs),
                            self.version_match(candidate.constraints, docs)]
                quality = _ranking_scores(docs, relevant, self.top_k)
                mismatch = _version_mismatch_rate(original.versions, docs)
                target = .45*quality["recall"] + .45*quality["mrr"] - .10*mismatch
                feature_rows.append(features); targets.append(target)
        self.ranker.fit(feature_rows, targets)


def reciprocal_rank_fusion(system: VACMQR, queries: List[str], k: int) -> List[Document]:
    scores, documents = {}, {}
    for query in queries:
        for rank, document in enumerate(system.retriever.search(query, k), 1):
            scores[document.id] = scores.get(document.id, 0.0) + 1/(60+rank)
            documents[document.id] = document
    return [documents[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]


def _ranking_scores(documents: List[Document], relevant_ids: set[str], k: int) -> Dict[str, float]:
    documents = documents[:k]
    flags = [document.id in relevant_ids for document in documents]
    recall = sum(flags)/max(len(relevant_ids), 1)
    precision = sum(flags)/max(len(documents), 1)
    mrr = next((1/rank for rank, hit in enumerate(flags, 1) if hit), 0.0)
    dcg = sum(hit/math.log2(rank+1) for rank, hit in enumerate(flags, 1))
    ideal = sum(1/math.log2(rank+1)
                for rank in range(1, min(len(relevant_ids), k)+1))
    return {"recall": recall, "precision": precision, "mrr": mrr,
            "ndcg": dcg/ideal if ideal else 0.0}


def _declared_versions(document: Document) -> List[str]:
    value = document.metadata.get("version", "")
    return re.findall(r"\d+(?:\.\d+){0,2}(?:[-._][a-z0-9]+)?", value.lower())


def _version_mismatch_rate(expected: List[str], documents: List[Document]) -> float:
    """Mismatch among documents that explicitly declare a version."""
    if not expected:
        return 0.0
    declared = [_declared_versions(document) for document in documents]
    comparable = [versions for versions in declared if versions]
    if not comparable:
        return 0.0
    expected_lower = {version.lower() for version in expected}
    mismatches = sum(not expected_lower.intersection(versions) for versions in comparable)
    return mismatches/len(comparable)


def _software_entity_coverage(cons: SEConstraints, documents: List[Document]) -> float:
    entities = list(dict.fromkeys(cons.versions + cons.apis + cons.errors
                                  + cons.frameworks + cons.languages + cons.files))
    if not entities:
        return 1.0
    context = " ".join(document.text + " " + " ".join(document.metadata.values())
                       for document in documents).lower()
    def present(entity: str) -> bool:
        canonical = re.sub(r"\(\)$", "", entity.lower())
        return canonical in context
    return sum(present(entity) for entity in entities)/len(entities)


def _repair_evidence_hit(example: QAExample, documents: List[Document]) -> float | None:
    if not example.repair_doc_ids:
        return None
    retrieved = {document.id for document in documents}
    return float(bool(retrieved.intersection(example.repair_doc_ids)))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _bootstrap_delta_ci(deltas: Sequence[float], samples: int = 2000,
                        seed: int = 42) -> Tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    array = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    low, high = np.quantile(means, [.025, .975])
    return float(low), float(high)


def evaluate_rewrite_constraints(results: Sequence[VACMQRResult]) -> Dict[str, float]:
    """Aggregate constraint preservation over generated (non-original) candidates."""
    details = [item for result in results for item in result.filter_details]
    if not details:
        return {"VPR": 1.0, "APR": 1.0, "EPR": 1.0,
                "IntentPreservation": 1.0, "CandidateAcceptanceRate": 1.0}
    return {
        "VPR": round(_mean([item["version"] for item in details]), 4),
        "APR": round(_mean([item["api"] for item in details]), 4),
        "EPR": round(_mean([item["error"] for item in details]), 4),
        "IntentPreservation": round(_mean([item["intent"] for item in details]), 4),
        "CandidateAcceptanceRate": round(_mean([float(item["kept"]) for item in details]), 4),
    }


def aggregate_answer_evaluations(records: Sequence[AnswerEvaluationRecord]) -> Dict[str, float]:
    """Aggregate externally verified answer metrics without inventing judge labels."""
    if not records:
        return {}
    output = {
        "AnswerCorrectness": _mean([r.correctness for r in records]),
        "Faithfulness": _mean([r.faithfulness for r in records]),
        "AnswerRelevance": _mean([r.answer_relevance for r in records]),
        "CitationPrecision": _mean([r.citation_precision for r in records]),
        "CitationRecall": _mean([r.citation_recall for r in records]),
    }
    fixes = [r.fix_success for r in records if r.fix_success is not None]
    if fixes:
        output["FixSuccessRate"] = _mean(fixes)
    return {name: round(value, 4) for name, value in output.items()}


def evaluate_conformal_trigger(system: VACMQR,
                               examples: Sequence[TriggerCalibrationExample]) -> Dict[str, float]:
    """Evaluate trigger coverage on a held-out set, not the calibration set."""
    if system.conformal_trigger.quantile is None:
        raise RuntimeError("calibrate_trigger() must be called first")
    covered, singleton, false_singleton_trigger = [], [], []
    set_sizes = []
    for example in examples:
        cons = system.analyzer.extract_constraints(example.query)
        probability = system.uncertainty(
            example.query, system.analyzer.classification_entropy(example.query), cons)
        prediction = system.conformal_trigger.predict(probability)
        label = int(example.needs_rewrite)
        covered.append(float(label in prediction))
        singleton.append(float(len(prediction) == 1))
        false_singleton_trigger.append(float(prediction == [1] and label == 0))
        set_sizes.append(float(len(prediction)))
    return {"MarginalCoverage": round(_mean(covered), 4),
            "SingletonRate": round(_mean(singleton), 4),
            "FalseSingletonTriggerRate": round(_mean(false_singleton_trigger), 4),
            "AverageSetSize": round(_mean(set_sizes), 4),
            "TargetCoverage": round(1-system.conformal_trigger.alpha, 4)}


def evaluate(system: VACMQR, examples: Sequence[QAExample], k: int = 5) -> Dict[str, float]:
    return evaluate_comprehensive(system, examples, k)["vac_mqr_retrieval"]


def evaluate_comprehensive(system: VACMQR, examples: Sequence[QAExample],
                           k: int = 5, bootstrap_samples: int = 2000) -> Dict[str, Dict]:
    """Evaluate VAC-MQR, original-query baseline, robustness, and constraints."""
    vac_rows, base_rows, results = [], [], []
    version_mismatch, entity_coverage, repair_hits = [], [], []
    group_rows: Dict[str, List[float]] = {}
    paired_robustness: Dict[str, Dict[str, List[float]]] = {}
    latencies_ms: List[float] = []

    for example in examples:
        started = time.perf_counter()
        result = system.run(example.query)
        results.append(result)
        vac_docs = reciprocal_rank_fusion(system, result.final_queries, k)
        latencies_ms.append((time.perf_counter()-started)*1000)
        base_docs = system.retriever.search(example.query, k)
        relevant = set(example.relevant_doc_ids)
        vac = _ranking_scores(vac_docs, relevant, k)
        base = _ranking_scores(base_docs, relevant, k)
        vac_rows.append(vac); base_rows.append(base)
        cons = system.analyzer.extract_constraints(example.query)
        version_mismatch.append(_version_mismatch_rate(cons.versions, vac_docs))
        entity_coverage.append(_software_entity_coverage(cons, vac_docs))
        repair_hit = _repair_evidence_hit(example, vac_docs)
        if repair_hit is not None:
            repair_hits.append(repair_hit)
        group_rows.setdefault(example.group, []).append(vac["recall"])
        if example.pair_id:
            entry = paired_robustness.setdefault(example.pair_id, {})
            entry.setdefault(example.group, []).append(vac["recall"])

    def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
        return {f"Recall@{k}": round(_mean([r["recall"] for r in rows]), 4),
                f"Precision@{k}": round(_mean([r["precision"] for r in rows]), 4),
                "MRR": round(_mean([r["mrr"] for r in rows]), 4),
                f"nDCG@{k}": round(_mean([r["ndcg"] for r in rows]), 4)}

    recall_deltas = [v["recall"]-b["recall"] for v, b in zip(vac_rows, base_rows)]
    mrr_deltas = [v["mrr"]-b["mrr"] for v, b in zip(vac_rows, base_rows)]
    recall_ci = _bootstrap_delta_ci(recall_deltas, bootstrap_samples)
    mrr_ci = _bootstrap_delta_ci(mrr_deltas, bootstrap_samples)
    no_harm = _mean([float(v["recall"] >= b["recall"] and v["mrr"] >= b["mrr"])
                     for v, b in zip(vac_rows, base_rows)])

    domain = {f"VMR@{k}": round(_mean(version_mismatch), 4),
              f"SEC@{k}": round(_mean(entity_coverage), 4),
              "NoHarmRate": round(no_harm, 4)}
    if repair_hits:
        domain[f"FER@{k}"] = round(_mean(repair_hits), 4)

    robustness_degradation = []
    for groups in paired_robustness.values():
        if "clean" not in groups:
            continue
        clean = _mean(groups["clean"])
        for group, values in groups.items():
            if group != "clean":
                robustness_degradation.append((clean-_mean(values))/max(clean, 1e-12))

    return {
        "vac_mqr_retrieval": aggregate(vac_rows),
        "original_retrieval": aggregate(base_rows),
        "software_engineering": domain,
        "rewrite_constraints": evaluate_rewrite_constraints(results),
        "system_behavior": {
            "RewriteTriggerRate": round(_mean([float(r.rewrite_triggered) for r in results]), 4),
            "AverageCandidateCount": round(_mean([float(len(r.candidates)) for r in results]), 4),
            "FusionRate": round(_mean([float(r.strategy == "fused") for r in results]), 4),
            "AverageEndToEndLatencyMs": round(_mean(latencies_ms), 3),
        },
        "paired_improvement": {
            f"DeltaRecall@{k}": round(_mean(recall_deltas), 4),
            "DeltaRecall95CI": tuple(round(x, 4) for x in recall_ci),
            "DeltaMRR": round(_mean(mrr_deltas), 4),
            "DeltaMRR95CI": tuple(round(x, 4) for x in mrr_ci),
        },
        "group_recall": {group: round(_mean(values), 4)
                         for group, values in group_rows.items()},
        "robustness": {
            "PairedRobustnessDegradation": (round(_mean(robustness_degradation), 4)
                                              if robustness_degradation else None),
            "ComparedPairCount": len(robustness_degradation),
        },
    }


def show(result: VACMQRResult) -> None:
    print("="*88)
    print(f"query={result.query}\nintent={result.intent} uncertainty={result.uncertainty} "
          f"triggered={result.rewrite_triggered} conformal_set={result.trigger_prediction_set}")
    print(f"versions={result.constraints.versions} apis={result.constraints.apis} "
          f"errors={result.constraints.errors} frameworks={result.constraints.frameworks}")
    for item in result.filter_details:
        state = "KEEP" if item["kept"] else "DROP"
        print(f"[{state}] {item['view']:<12} G={item['score']:.3f} "
              f"V={item['version']:.1f} API={item['api']:.1f} ERR={item['error']:.1f} "
              f"{item['candidate']}")
    for item in result.retrieval_details:
        print(f"[RANK] {item['view']:<12} S={item['proxy_score']:.3f} "
              f"VM={item['version_match']:.2f} SC={item['symbol_coverage']:.2f} "
              f"docs={item['doc_ids']}")
    print(f"strategy={result.strategy} final={result.final_queries}")


def main() -> None:
    corpus = [
        Document("d1", "Spring Boot 3.2 RedisTemplate SerializationException is fixed by configuring GenericJackson2JsonRedisSerializer.", {"version": "3.2"}),
        Document("d2", "Spring Boot 2.7 RedisTemplate uses JdkSerializationRedisSerializer by default.", {"version": "2.7"}),
        Document("d3", "Python 3.11 pandas.read_csv raises UnicodeDecodeError when encoding differs; pass encoding='gbk' for a GBK file.", {"version": "3.11"}),
        Document("d4", "Python 3.9 pandas.read_excel API usage and engine configuration.", {"version": "3.9"}),
        Document("d5", "React 18 useEffect cleanup prevents updates after a component is unmounted.", {"version": "18"}),
        Document("d6", "React 17 class component lifecycle migration guide.", {"version": "17"}),
    ]
    examples = [
        QAExample("Spring Boot 3.2 RedisTemplate 报 SerializationException 咋办？",
                  ["d1"], repair_doc_ids=["d1"], group="colloquial", pair_id="spring-1"),
        QAExample("Spring Boot 3.2 RedisTemplate 的 SerializationException 如何解决？",
                  ["d1"], repair_doc_ids=["d1"], group="clean", pair_id="spring-1"),
        QAExample("Python 3.11 pandas.read_csv() 出现 UnicodeDecodeError 如何解决？",
                  ["d3"], repair_doc_ids=["d3"], group="clean"),
        QAExample("React 18 useEffect() 清理函数有什么作用？",
                  ["d5"], group="clean"),
    ]
    embedder = HashEmbedder()
    system = VACMQR(SEQueryAnalyzer(), embedder, EmbeddingRetriever(corpus, embedder))
    for example in examples:
        show(system.run(example.query))
    print("comprehensive_evaluation=")
    for section, metrics in evaluate_comprehensive(system, examples, k=3).items():
        print(f"  {section}: {metrics}")


if __name__ == "__main__":
    main()
