import argparse
import sys
import os
import joblib
import pandas as pd

from stages.stage1 import MyStage1Model
from stages.stage2 import MyStage2Model

ASSET_DISCARDED = "ASSET_DISCARDED"
INFRINGEMENT_CONFIRMED = "INFRINGEMENT_CONFIRMED"
INFRINGEMENT_DISCARDED = "INFRINGEMENT_DISCARDED"


def load_inputs(input_file, text_column):
    df = pd.read_csv(input_file, sep=None, engine="python")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {input_file}")

    texts = df[text_column].fillna("").astype(str).tolist()
    return df, texts


def predict_pipeline(texts, stage1, stage2):
    outputs = []

    stage1_pred = stage1.predict(texts)

    texts_s2 = []
    idx_s2 = []

    for i, pred in enumerate(stage1_pred):
        if pred == 0:
            outputs.append(ASSET_DISCARDED)
        else:
            outputs.append(None)
            texts_s2.append(texts[i])
            idx_s2.append(i)

    if texts_s2:
        stage2_pred = stage2.predict(texts_s2)

        for i, pred in zip(idx_s2, stage2_pred):
            outputs[i] = INFRINGEMENT_CONFIRMED if pred == 1 else INFRINGEMENT_DISCARDED

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="CSV with titles to analyze")
    parser.add_argument("--text_column", type=str, default="title", help="Column containing the text")
    parser.add_argument("--stage1_model", type=str, default="models/stage1.joblib")
    parser.add_argument("--stage2_model", type=str, default="models/stage2.joblib")
    parser.add_argument("--output_file", type=str, default="outputs/inference_results.csv")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.stage1_model):
            raise FileNotFoundError(f"Stage1 model not found: {args.stage1_model}")

        if not os.path.exists(args.stage2_model):
            raise FileNotFoundError(f"Stage2 model not found: {args.stage2_model}")

        stage1 = joblib.load(args.stage1_model)
        stage2 = joblib.load(args.stage2_model)

        df, texts = load_inputs(args.input_file, args.text_column)
        final_labels = predict_pipeline(texts, stage1, stage2)

        df["final_label"] = final_labels

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        df.to_csv(args.output_file, index=False)

        print(df["final_label"].tolist())
        print(f"\nSaved to: {args.output_file}")

    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("[error] Empty CSV")
        sys.exit(1)
    except Exception as e:
        print(f"[error] Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
