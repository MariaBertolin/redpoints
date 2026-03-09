import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from src.dataset import MyDataset
from src.stages.stage1 import MyStage1Model

try:
    from src.stages.stage2 import MyStage2Model
except ImportError:
    MyStage2Model = None


def default_stage1_binarize(labels):
    s = pd.Series(labels)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0, 1}):
        return numeric.astype(int)
    return (s.astype(str).str.upper() != "ASSET_DISCARDED").astype(int)


class PipelineService:
    def __init__(
        self,
        dataset_path: str = "data/reference_listing.csv",
        stage1_model_path: str = "models/stage1.pkl",
        stage2_model_path: str = "models/stage2.pkl",
        metadata_path: str = "models/metadata.json",
        stage1_threshold: float = 0.20,
        stage2_threshold: float = 0.65,
        max_iter: int = 1000,
    ):
        self.dataset_path = dataset_path
        self.stage1_model_path = Path(stage1_model_path)
        self.stage2_model_path = Path(stage2_model_path)
        self.metadata_path = Path(metadata_path)
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.max_iter = max_iter

        self.stage1 = None
        self.stage2 = None
        self.reference_texts: list[str] = []
        self.reference_vectors = None

        self.metadata = {
            "model_paths": {
                "stage1_model": str(self.stage1_model_path),
                "stage2_model": str(self.stage2_model_path),
                "reference_embeddings": "computed_in_memory_from_stage1_vectorizer",
            },
            "datasets": {
                "stage1": {
                    "train": dataset_path,
                    "validation": dataset_path,
                },
                "stage2": {
                    "train": "not_available",
                    "validation": "not_available",
                },
            },
            "metrics": {},
        }

        self._load_or_train()

    def _load_or_train(self) -> None:
        dataset = MyDataset(self.dataset_path)
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.split()

        self.reference_texts = list(X_train)

        if self.stage1_model_path.exists():
            self.stage1 = joblib.load(self.stage1_model_path)
        else:
            self.stage1 = MyStage1Model(self.max_iter, self.stage1_threshold)
            self.stage1.fit(list(X_train), list(y_train))
            self.stage1_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.stage1, self.stage1_model_path)

        self.reference_vectors = self.stage1.vectorizer.transform(self.reference_texts)
        self.metadata["metrics"]["stage1"] = self._compute_stage1_metrics(X_val, y_val)

        if self.stage2_model_path.exists():
            self.stage2 = joblib.load(self.stage2_model_path)
            self.metadata["metrics"]["stage2"] = {
                "precision": None,
                "recall": None,
                "f1": None,
            }
        elif MyStage2Model is not None:
            self.stage2 = None
            self.metadata["metrics"]["stage2"] = {
                "precision": None,
                "recall": None,
                "f1": None,
            }
        else:
            self.stage2 = None
            self.metadata["metrics"]["stage2"] = {
                "precision": None,
                "recall": None,
                "f1": None,
            }

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _compute_stage1_metrics(self, X_val, y_val) -> dict[str, float]:
        y_true = default_stage1_binarize(y_val)
        y_pred = self.stage1.predict(list(X_val))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def _similarity_search(self, title: str, top_k: int) -> tuple[float, list[dict[str, Any]]]:
        query_vec = self.stage1.vectorizer.transform([title])
        sims = cosine_similarity(query_vec, self.reference_vectors).flatten()

        if len(sims) == 0:
            return 0.0, []

        top_k = min(top_k, len(self.reference_texts))
        top_idx = sims.argsort()[::-1][:top_k]

        matches = []
        for rank_idx in top_idx:
            matches.append(
                {
                    "reference_id": int(rank_idx),
                    "title": self.reference_texts[rank_idx],
                    "score": float(sims[rank_idx]),
                }
            )

        best_score = float(matches[0]["score"]) if matches else 0.0
        return best_score, matches

    def _run_stage2(self, title: str) -> dict[str, Any]:
        if self.stage2 is None:
            return {
                "executed": False,
                "score": None,
                "threshold": self.stage2_threshold,
                "suspicion_flag": None,
            }

        if hasattr(self.stage2, "predict_proba"):
            proba = self.stage2.predict_proba([title])
            first = proba[0]
            score = float(first[1]) if hasattr(first, "__len__") else float(first)
        else:
            pred = self.stage2.predict([title])
            score = float(pred[0])

        return {
            "executed": True,
            "score": score,
            "threshold": self.stage2_threshold,
            "suspicion_flag": score >= self.stage2_threshold,
        }

    def analyze(self, title: str, top_k: int) -> dict[str, Any]:
        proba = self.stage1.predict_proba([title])
        first = proba[0]
        stage1_score = float(first[1]) if hasattr(first, "__len__") else float(first)

        is_asset = stage1_score >= self.stage1_threshold
        similarity_score, matches = self._similarity_search(title, top_k)

        if is_asset:
            stage2_result = self._run_stage2(title)
        else:
            stage2_result = {
                "executed": False,
                "score": None,
                "threshold": self.stage2_threshold,
                "suspicion_flag": None,
            }

        return {
            "title": title,
            "is_asset": is_asset,
            "stage1": {
                "score": stage1_score,
                "threshold": self.stage1_threshold,
            },
            "similarity": {
                "score": similarity_score,
                "top_k": top_k,
                "matches": matches,
            },
            "stage2": stage2_result,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_metadata(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self.metadata