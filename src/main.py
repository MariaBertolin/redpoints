import argparse
import sys
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

from dataset import MyDataset
from stages.stage1 import MyStage1Model
from stages.stage2 import MyStage2Model
from stages.similarity import find_similar_listings, build_reference_matrix
from metrics import compute_metrics, evaluate_thresholds
from utils import binarize_stage1, binarize_stage2, build_vectorizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="Dataset path", type=str, default="data/reference_listing.csv")
    parser.add_argument("--max_iter", help="Logistic Regression max_iter", type=int, default=1000)
    parser.add_argument("--stage1_threshold", type=float, default=0.20)
    parser.add_argument("--stage2_threshold", type=float, default=0.25)
    parser.add_argument("--stage2_sim_threshold", type=float, default=0.85)
    parser.add_argument("--similarity_query", help="Text to compute similarity", type=str, default=None)
    parser.add_argument("--similarity_topk", help="Number of Top-k similar listings", type=int, default=5)
    parser.add_argument("--run_stage1", help="Execute stage1", action="store_true")
    parser.add_argument("--run_similarity", help="Run similarity", action="store_true")
    parser.add_argument("--run_stage2", help="Execute stage2", action="store_true")
    args = parser.parse_args()

    try:
        # Dataset
        # # Divisió del dataset en 70% train - 15% val - 15% test        
        dataset = MyDataset(args.dataset_path)
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.split()

        print("Samples dimension stage1:")
        print(f"  train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

        ## STAGE 1
        if args.run_stage1:
            print("\n=== Stage1 results ===")
            stage1 = MyStage1Model(args.max_iter, args.stage1_threshold)
            stage1.fit(list(X_train), list(y_train))
            y_val_pred = stage1.predict(list(X_val))
            y_test_pred = stage1.predict(list(X_test))

            try:
                y_val_proba = stage1.predict_proba(list(X_val))
                y_test_proba = stage1.predict_proba(list(X_test))
            except AttributeError:
                y_val_proba = None
                y_test_proba = None

            val_m = compute_metrics(y_val.apply(binarize_stage1), y_val_pred, y_val_proba)
            test_m = compute_metrics(y_test.apply(binarize_stage1), y_test_pred, y_test_proba)
            
            metrics_df = pd.DataFrame([val_m, test_m], index=["val", "test"])
            print(metrics_df)
        
        ## SIMILARITY
        if args.run_similarity:
            print("\n=== Similarity analysis ===")

            reference_texts = list(X_train) + list(X_val) + list(X_test)
            sample_texts = list(X_test)

            if len(reference_texts) == 0:
                print("No reference texts available.")
            else:
                vectorizer = build_vectorizer()
                vectorizer.fit(reference_texts)

                query_text = args.similarity_query
                if not query_text:
                    query_text = sample_texts[50] if len(sample_texts) > 0 else reference_texts[0]

                k = min(args.similarity_topk, len(reference_texts))

                reference_texts, reference_vectors = build_reference_matrix(
                    reference_texts,
                    vectorizer
                )

                sim_res = find_similar_listings(
                    query_text,
                    reference_texts,
                    reference_vectors,
                    vectorizer,
                    k=k
                )

                print(f"query: {sim_res['query']}")
                for item in sim_res["similar"]:
                    print(f"  [{item['similarity']:.3f}] {item['text']}")

        if args.run_stage2:
            # Em quedo amb les files classificades com a asset
            train_asset_mask = stage1.predict(list(X_train)) == 1
            X_train_s2 = X_train[train_asset_mask]
            y_train_s2 = y_train[train_asset_mask]

            val_asset_mask = stage1.predict(list(X_val)) == 1
            X_val_s2 = X_val[val_asset_mask]
            y_val_s2 = y_val[val_asset_mask]

            test_asset_mask = stage1.predict(list(X_test)) == 1
            X_test_s2 = X_test[test_asset_mask]
            y_test_s2 = y_test[test_asset_mask]

            print(f"\nSamples dimension stage2 (after stage 1 classification):")
            print(f"  train: {len(X_train_s2)} | val: {len(X_val_s2)} | test: {len(X_test_s2)}")
            print("\n=== Stage2 results ===")
            if len(X_train_s2) == 0:
                raise ValueError("No training samples reached stage2.")

            stage2 = MyStage2Model(
                max_iter=args.max_iter,
                threshold=args.stage2_threshold,
                sim_threshold=args.stage2_sim_threshold
            )
            stage2.fit(list(X_train_s2), list(y_train_s2))

            # VAL
            if len(X_val_s2) > 0:
                y_val_true_s2 = binarize_stage2(y_val_s2)
                y_val_proba_s2 = stage2.predict_proba(list(X_val_s2))
                y_val_pred_s2 = stage2.predict(list(X_val_s2))

                val_m_s2 = compute_metrics(y_val_true_s2, y_val_pred_s2, y_val_proba_s2)

                """ print("\nStage2 threshold sweep (val):")
                thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                th_df = evaluate_thresholds(y_val_true_s2, y_val_proba_s2, thresholds)
                print(th_df) """
            else:
                print("\nNo validation samples reached stage2.")

            # TEST
            if len(X_test_s2) > 0:
                y_test_true_s2 = binarize_stage2(y_test_s2)
                y_test_proba_s2 = stage2.predict_proba(list(X_test_s2))
                y_test_pred_s2 = stage2.predict(list(X_test_s2))

                test_m_s2 = compute_metrics(y_test_true_s2, y_test_pred_s2, y_test_proba_s2)
            else:
                print("\nNo test samples reached stage2.")
            
            metrics_df = pd.DataFrame([val_m_s2, test_m_s2], index=["val", "test"])
            print(metrics_df)

    except FileNotFoundError:
        print(f"[error] File not found: {args.dataset_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("[error] Empty CSV")
        sys.exit(1)
    except Exception as e:
        print(f"[error] Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
