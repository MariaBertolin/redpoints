from sklearn.linear_model import LogisticRegression
from utils import build_vectorizer, binarize_stage1

class MyStage1Model:
    """
    Stage1Model: A text classification model using TF-IDF vectorization and Logistic Regression.
    
    This class implements a text classifier that combines character-level TF-IDF vectorization
    with logistic regression for binary or multi-class text classification tasks.
    
    Attributes:
        vectorizer (TfidfVectorizer): Converts text to character n-gram TF-IDF features.
        classifier (LogisticRegression): Binary/multi-class logistic regression classifier.
        is_fitted (bool): Flag indicating whether the model has been trained.
    
    Example:git
        model = Stage1Model()
        model.fit(training_texts, training_labels)
        predictions = model.predict(test_texts)
    """
    
    def __init__(self, max_iter=1000, threshold=0.2):
        # Careful! A tenir en compte: tenim files amb diferents idiomes!!
        # -> Matriu de features: tot minúscules, sense accents, n-grams de caràcters de 3 a 5
        self.vectorizer = build_vectorizer()
        # Classificador binari BÀSIC amb L2 regularization per defecte!
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=42, # número fix per reproducibilitat (de Hitchhiker's Guide to the Galaxy:)
            class_weight="balanced" # important perquè hi ha poques mostres de ASSET_DISCARDED
        )
        self.threshold = threshold
        self.is_fitted = False
    
    #TRAINING
    def fit(self, texts, labels):
        # stage 1 es només binaria la classificació! tot el que és ASSET_DISCARDED(4) = 0
        labels = binarize_stage1(labels)
        # Train simple, vectoritzem i entrenem el logistic regression
        if len(texts) == 0 or len(labels) == 0:
            raise ValueError("texts & labels sense data!")
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True
    
    #PREDICTION: Classes estimades (0 o 1) segons el threshold
    def predict(self, texts, threshold=None):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        if threshold is None:
            threshold = self.threshold

        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)

    #PREDICTION: Probabilitats estimades per a la classe positiva (1), de cada sample
    def predict_proba(self, texts):
        # Return probability estimates for predictions
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
