import argparse
import json
import re

import fasttext
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()  # limpiar espacios extra
    text = re.sub(r"([.,!?;:()\[\]{}])", r" \1 ", text)  # separar puntuación
    return text


def load_data(json_path: str, threshold: float, label: str, balance_method: str = None):
    """
    Carga los datos desde un archivo JSON y los prepara en archivos de texto
    para ser utilizados por fastText, pudiendo balancear las clases en el
    conjunto de entrenamiento mediante undersampling u oversampling.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = [x.get("original_text", "") for x in data]
    labels = []
    for x in data:
        modCategories = x.get("moderation_result", {}).get("moderationCategories", [])
        category = next(
            (cat for cat in modCategories if cat.get("name") == label),
            None,
        )
        confidence = category.get("confidence", 0.0) if category else 0.0
        labels.append("__label__1" if confidence > threshold else "__label__0")

    df = pd.DataFrame({"text": texts, "label": labels})

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    if balance_method:
        class_0 = train_df[train_df["label"] == "__label__0"]
        class_1 = train_df[train_df["label"] == "__label__1"]
        if balance_method == "undersample" and len(class_0) and len(class_1):
            if len(class_0) > len(class_1):
                class_0 = resample(
                    class_0, replace=False, n_samples=len(class_1), random_state=42
                )
            else:
                class_1 = resample(
                    class_1, replace=False, n_samples=len(class_0), random_state=42
                )
        elif balance_method == "oversample" and len(class_0) and len(class_1):
            if len(class_0) < len(class_1):
                class_0 = resample(
                    class_0, replace=True, n_samples=len(class_1), random_state=42
                )
            else:
                class_1 = resample(
                    class_1, replace=True, n_samples=len(class_0), random_state=42
                )
        train_df = pd.concat([class_0, class_1])
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def save_fasttext_data(df, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                text = row.get("text", "").replace("_usr", "").replace("_user", "")
                text = clean_text(text)
                if text:
                    f.write(f"{row['label']} {text}\n")

    save_fasttext_data(train_df, "train.txt")
    save_fasttext_data(val_df, "valid.txt")
    save_fasttext_data(test_df, "test.txt")


def evaluate_model(model: fasttext.FastText, test_file: str, model_name: str):
    # Leer líneas válidas: debe haber etiqueta y texto
    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            label, text = parts
            text = text.strip()
            if not text:
                continue
            records.append((label, text))

    if not records:
        print("⚠ No hay datos válidos para evaluación.")
        return

    test_df = pd.DataFrame(records, columns=["label", "text"])
    labels_true = [1 if lbl == "__label__1" else 0 for lbl in test_df["label"]]

    # Predecir todos a la vez (FastText acepta lista)
    texts = test_df["text"].tolist()
    preds, _ = model.predict(texts)
    labels_pred = [1 if p[0] == "__label__1" else 0 for p in preds]

    acc = accuracy_score(labels_true, labels_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_true, labels_pred, average="binary"
    )
    cm = confusion_matrix(labels_true, labels_pred)

    metrics = (
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall: {rec:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    print(metrics)
    with open(f"{model_name}_metrics.txt", "w", encoding="utf-8") as out:
        out.write(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="FastText toxicity classifier trainer with embeddings"
    )
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--autotuneDuration", type=int, default=0)
    parser.add_argument("--word_ngrams", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--minCount", type=int)
    parser.add_argument(
        "--loss", type=str, default="softmax", help="Función de pérdida"
    )
    parser.add_argument(
        "--minn", type=int, help="Tamaño mínimo de n-gramas de caracteres"
    )
    parser.add_argument(
        "--maxn", type=int, help="Tamaño máximo de n-gramas de caracteres"
    )
    parser.add_argument(
        "--bucket", type=int, help="Número de buckets para hashing de n-gramas"
    )
    parser.add_argument(
        "--neg", type=int, help="Número de muestras negativas para negative sampling"
    )
    parser.add_argument(
        "--pretrained_vectors",
        type=str,
        help="Path to pretrained word vectors (.vec subset).",
        default=None,
    )
    parser.add_argument(
        "--dim", type=int, help="Dimensionality of pretrained vectors.", default=None
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--undersample", action="store_true")
    group.add_argument("--oversample", action="store_true")

    args = parser.parse_args()
    if args.undersample:
        balance = "undersample"
    elif args.oversample:
        balance = "oversample"
    else:
        balance = None

    load_data(args.json_path, args.threshold, args.label, balance)

    print("Entrenando modelo...")
    train_args = {"input": "train.txt"}
    if args.autotuneDuration > 0:
        train_args.update(
            {
                "autotuneValidationFile": "valid.txt",
                "autotuneDuration": args.autotuneDuration,
                "autotuneMetric": "f1:__label__1",
            }
        )
    if args.word_ngrams is not None:
        train_args["wordNgrams"] = args.word_ngrams
    if args.epoch is not None:
        train_args["epoch"] = args.epoch
    if args.lr is not None:
        train_args["lr"] = args.lr
    if args.pretrained_vectors:
        train_args["pretrainedVectors"] = args.pretrained_vectors
    if args.dim:
        train_args["dim"] = args.dim
    if args.minCount:
        train_args["minCount"] = args.minCount
    if args.loss:
        train_args["loss"] = args.loss
    if args.minn is not None:
        train_args["minn"] = args.minn
    if args.maxn is not None:
        train_args["maxn"] = args.maxn
    if args.bucket is not None:
        train_args["bucket"] = args.bucket
    if args.neg is not None:
        train_args["neg"] = args.neg

    train_args["verbose"] = 2
    model = fasttext.train_supervised(**train_args)
    model.save_model(args.output_model)

    print("Evaluando modelo...")
    evaluate_model(model, "test.txt", args.output_model)


if __name__ == "__main__":
    main()
