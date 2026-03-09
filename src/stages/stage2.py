import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import build_vectorizer, binarize_stage2
from stages.similarity import build_reference_matrix, find_similar_listings


class MyStage2Model:
    def __init__(self, max_iter=1000, threshold=0.35, sim_threshold=0.85):
        self.threshold = threshold
        self.sim_threshold = sim_threshold

        self.vectorizer = build_vectorizer()
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            class_weight="balanced"
        )

        self.reference_texts = []
        self.reference_vectors = None
        self.is_fitted = False

    def fit(self, texts, labels):
        texts = pd.Series(texts).astype(str).reset_index(drop=True)
        y = binarize_stage2(labels)

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)

        self.reference_texts = texts[y == 1].tolist()
        self.reference_texts, self.reference_vectors = build_reference_matrix(
            self.reference_texts,
            self.vectorizer
        )

        self.is_fitted = True

    def predict_proba(self, texts):
        if not self.is_fitted:
            raise ValueError("Stage2 model not fitted.")

        texts = pd.Series(texts).astype(str)
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)[:, 1]

    def predict_similarity(self, texts):
        if not self.is_fitted:
            raise ValueError("Stage2 model not fitted.")

        texts = pd.Series(texts).astype(str).tolist()
        scores = []

        for text in texts:
            result = find_similar_listings(
                query_text=text,
                reference_texts=self.reference_texts,
                reference_vectors=self.reference_vectors,
                vectorizer=self.vectorizer,
                k=1
            )

            if result["similar"]:
                top_score = result["similar"][0].get("score", result["similar"][0].get("similarity", 0.0))
            else:
                top_score = 0.0

            scores.append(top_score)

        return scores

    def predict(self, texts):
        lr_scores = self.predict_proba(texts)
        sim_scores = self.predict_similarity(texts)

        preds = []
        for lr, sim in zip(lr_scores, sim_scores):
            pred = int((lr >= self.threshold) or (sim >= self.sim_threshold))
            preds.append(pred)

        return preds

    def forward(self, x):
        return self.predict(x)

    def explain(self, text, topk=5):
        if not self.is_fitted:
            raise ValueError("Stage2 model not fitted.")

        return find_similar_listings(
            query_text=str(text),
            reference_texts=self.reference_texts,
            reference_vectors=self.reference_vectors,
            vectorizer=self.vectorizer,
            k=topk
        )