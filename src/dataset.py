import pandas as pd
from sklearn.model_selection import train_test_split

class MyDataset:
    
    def __init__(self, dataset_path):
        df = pd.read_csv(dataset_path)
        self.titles = df.iloc[:, 0].astype(str)
        self.labels = df.iloc[:, 1]
        # stage 1 es només binaria la classificació! tot el que és ASSET_DISCARDED(4) = 0
        #   self.labels = (labels != 4).astype(int)

    def split(self, random_state=42):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.titles,
            self.labels,
            # 15% del dataset per a test, 85% per a train+val
            test_size=0.15,
            random_state=random_state,
            stratify=self.labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            # 17.6% del dataset per a val (17.6% de 85% és aproximadament el 15% del total), la resta train
            test_size=0.176,
            random_state=random_state,
            stratify=y_train_val
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
