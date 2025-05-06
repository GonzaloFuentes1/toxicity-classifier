
import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# === Environment setup ===
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def mark_positive_sample(labels, threshold=0.5):
    return int(any(score >= threshold for score in labels))


def undersample_data(df: pd.DataFrame, label_column: str, threshold: float = 0.5):
    df = df.copy()
    df["binary_labels"] = df[label_column].apply(
        lambda x: [1 if score >= threshold else 0 for score in x]
    )
    binarized = pd.DataFrame(df["binary_labels"].tolist())
    selected_indices = set()

    for col in binarized.columns:
        positives = binarized[binarized[col] == 1]
        negatives = binarized[binarized[col] == 0]
        min_class = min(len(positives), len(negatives))
        if min_class == 0:
            continue
        pos_indices = positives.index.tolist()
        neg_indices = negatives.sample(n=min_class, random_state=42).index.tolist()
        selected_indices.update(pos_indices + neg_indices)

    balanced_df = df.loc[list(selected_indices)].drop(columns=["binary_labels"])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


def oversample_data(df: pd.DataFrame, label_column: str, threshold: float = 0.5):
    df = df.copy()
    df["binary_labels"] = df[label_column].apply(
        lambda x: [1 if score >= threshold else 0 for score in x]
    )
    binarized = pd.DataFrame(df["binary_labels"].tolist())
    selected_indices = set()

    for col in binarized.columns:
        positives = binarized[binarized[col] == 1]
        negatives = binarized[binarized[col] == 0]
        max_class = max(len(positives), len(negatives))
        if len(positives) == 0 or len(negatives) == 0:
            continue
        if len(positives) < max_class:
            pos_indices = resample(
                positives, replace=True, n_samples=max_class, random_state=42
            ).index.tolist()
            neg_indices = negatives.index.tolist()
        else:
            pos_indices = positives.index.tolist()
            neg_indices = resample(
                negatives, replace=True, n_samples=max_class, random_state=42
            ).index.tolist()
        selected_indices.update(pos_indices + neg_indices)

    balanced_df = df.loc[list(selected_indices)].drop(columns=["binary_labels"])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


def load_data(
    json_path: str, label_ids: list, balance_method: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = [x["original_text"] for x in data]
    scores = [
        [
            float(x["moderation_result"]["moderationCategories"][id]["confidence"])
            for id in label_ids
        ]
        for x in data
    ]
    df = pd.DataFrame({"text": texts, "labels": scores})
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    if balance_method == "undersample":
        train_df = undersample_data(train_df, label_column="labels")
    elif balance_method == "oversample":
        train_df = oversample_data(train_df, label_column="labels")

    return train_df, val_df, test_df


def compute_metrics(eval_pred: Any, threshold: float) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    pred_labels = (probs >= threshold).astype(int)
    true_labels = (labels >= threshold).astype(int)

    y_true_bin = (true_labels.sum(axis=1) > 0).astype(int)
    y_pred_bin = (pred_labels.sum(axis=1) > 0).astype(int)

    f1_binary = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    precision_binary = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall_binary = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    f1_micro = f1_score(true_labels, pred_labels, average="micro", zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )

    per_class_metrics = {
        f"class_{i+1}_precision": p for i, p in enumerate(precision_class)
    }
    per_class_metrics.update({
        f"class_{i+1}_recall": r for i, r in enumerate(recall_class)
    })
    per_class_metrics.update({
        f"class_{i+1}_f1": f for i, f in enumerate(f1_class)
    })

    return {
        "binary_f1": f1_binary,
        "binary_precision": precision_binary,
        "binary_recall": recall_binary,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1": np.mean(f1_class),
        **per_class_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo multilabel classification"
    )
    parser.add_argument("--json_path", required=True, help="Ruta al archivo JSON")
    parser.add_argument(
        "--output_dir", required=True, help="Directorio base para guardar el modelo"
    )
    parser.add_argument(
        "--logging_dir",
        required=True,
        help="Directorio base para guardar los logs de entrenamiento",
    )
    parser.add_argument(
        "--model_name", required=True, help="Nombre del modelo pre-entrenado"
    )
    parser.add_argument(
        "--label_ids",
        nargs="+",
        type=int,
        required=True,
        help="Índices de los labels en moderationCategories",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold para convertir las predicciones en etiquetas binarias",
    )
    parser.add_argument(
        "--balance_method",
        choices=["undersample", "oversample"],
        help="Método de balanceo para el conjunto de entrenamiento",
    )
    args = parser.parse_args()

    balance_suffix = args.balance_method if args.balance_method else "no-balance"
    model_name_safe = args.model_name.replace("/", "-")
    model_dir_suffix = f"{model_name_safe}_{balance_suffix}"
    full_output_dir = os.path.join(args.output_dir, model_dir_suffix)
    full_logging_dir = os.path.join(args.logging_dir, model_dir_suffix)

    os.makedirs(full_output_dir, exist_ok=True)
    os.makedirs(full_logging_dir, exist_ok=True)

    train_df, val_df, test_df = load_data(
        args.json_path, args.label_ids, balance_method=args.balance_method
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(args.label_ids),
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )
    training_args = TrainingArguments(
        output_dir=full_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        learning_rate=4e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.0001,
        warmup_ratio=0.01,
        logging_dir=full_logging_dir,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, args.threshold),
    )

    trainer.train()

    print("\n--- Validation Set ---")
    val_results = trainer.evaluate()
    print(val_results)

    print("\n--- Test Set ---")
    test_results = trainer.predict(tokenized_datasets["test"])
    test_predictions = test_results.predictions
    test_labels = test_results.label_ids

    test_metrics = compute_metrics((test_predictions, test_labels), args.threshold)
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    test_metrics_output_path = os.path.join(full_output_dir, "test_metrics.json")
    with open(test_metrics_output_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"\nTest metrics saved to {test_metrics_output_path}")


if __name__ == "__main__":
    main()
