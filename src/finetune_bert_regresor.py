import argparse
import json
import os
import pandas as pd
from typing import Any, Dict, Tuple
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def load_data(json_path: str, label_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos desde un JSON y divide en entrenamiento, validación y prueba.
    Se extrae el texto y la confianza del label indicado por label_id.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = [x["original_text"] for x in data]
    scores = [float(x["moderation_result"]["moderationCategories"][label_id]["confidence"]) for x in data]

    df = pd.DataFrame({"text": texts, "label": scores})

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df


def undersample_dataframe(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Realiza undersampling: reduce la clase mayoritaria tomando muestras sin reemplazo.
    Se convierte la variable numérica en binaria usando el threshold.
    """
    df = df.copy()
    df["binary_label"] = (df["label"] >= threshold).astype(int)

    positives = df[df["binary_label"] == 1]
    negatives = df[df["binary_label"] == 0]

    if len(positives) == 0 or len(negatives) == 0:
        print("⚠ No se puede undersamplear porque alguna clase no tiene muestras.")
        return df.drop(columns=["binary_label"])

    # Se elimina aleatoriamente muestras de la clase mayoritaria (sin reemplazo)
    if len(negatives) > len(positives):
        negatives = resample(negatives, replace=False, n_samples=len(positives), random_state=42)
    else:
        positives = resample(positives, replace=False, n_samples=len(negatives), random_state=42)

    balanced_df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df.drop(columns=["binary_label"])


def oversample_dataframe(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Realiza oversampling: aumenta la clase minoritaria replicando muestras (con reemplazo)
    hasta igualar el número de muestras de la clase mayoritaria.
    Se convierte la variable numérica en binaria usando el threshold.
    """
    df = df.copy()
    df["binary_label"] = (df["label"] >= threshold).astype(int)

    positives = df[df["binary_label"] == 1]
    negatives = df[df["binary_label"] == 0]

    if len(positives) == 0 or len(negatives) == 0:
        print("⚠ No se puede oversamplear porque alguna clase no tiene muestras.")
        return df.drop(columns=["binary_label"])

    # Se replica la clase minoritaria (con reemplazo) para igualar la cantidad de la mayoritaria
    if len(positives) < len(negatives):
        positives = resample(positives, replace=True, n_samples=len(negatives), random_state=42)
    else:
        negatives = resample(negatives, replace=True, n_samples=len(positives), random_state=42)

    balanced_df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df.drop(columns=["binary_label"])


def compute_metrics(eval_pred: Any, threshold: float) -> Dict[str, float]:
    """
    Calcula métricas de regresión y clasificación a partir de las predicciones.
    Convierte las predicciones a etiquetas binaria usando el threshold dado.
    """
    predictions, labels = eval_pred
    predictions = predictions.reshape(-1)

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    pred_labels = (predictions >= threshold).astype(int)
    true_labels = (labels >= threshold).astype(int)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary")
    try:
        auc = roc_auc_score(true_labels, predictions)
    except Exception:
        auc = float('nan')

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de regresión + clasificación")
    parser.add_argument("--json_path", required=True, help="Ruta al archivo JSON con los datos")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida para guardar el modelo y métricas")
    parser.add_argument("--logging_dir", required=True, help="Directorio para guardar los logs de entrenamiento")
    parser.add_argument("--model_name", required=True, help="Nombre del modelo pre-entrenado de Hugging Face")
    parser.add_argument("--label_id", type=int, default=0, help="Índice del label en moderationCategories")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold para convertir la label en binaria")
    parser.add_argument("--undersample", action="store_true", help="Activar undersampling del set de entrenamiento")
    parser.add_argument("--oversample", action="store_true", help="Activar oversampling del set de entrenamiento")
    args = parser.parse_args()

    # Verificar que no se activen ambos métodos al mismo tiempo.
    if args.undersample and args.oversample:
        print("⚠ Debes elegir solo una estrategia: undersample o oversample, no ambas.", flush=True)
        exit(1)

    # ----- Data -----
    train_df, val_df, test_df = load_data(args.json_path, args.label_id)

    if args.undersample:
        print("⚡ Aplicando undersampling al set de entrenamiento")
        train_df = undersample_dataframe(train_df, args.threshold)
    elif args.oversample:
        print("⚡ Aplicando oversampling al set de entrenamiento")
        train_df = oversample_dataframe(train_df, args.threshold)
    else:
        print("⚡ Sin balanceo (undersample/oversample) aplicado al set de entrenamiento")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = 1  # Problema de regresión
    config.problem_type = "regression"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=15,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=args.logging_dir,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, args.threshold)
    )

    trainer.train()

    print("\n--- Validation Set ---")
    val_results = trainer.evaluate()
    print(val_results)

    print("\n--- Test Set ---")
    test_results = trainer.predict(tokenized_datasets["test"])
    test_predictions = test_results.predictions.reshape(-1)
    test_labels = test_results.label_ids

    pred_bin = (test_predictions >= args.threshold).astype(int)
    true_bin = (test_labels >= args.threshold).astype(int)

    cm = confusion_matrix(true_bin, pred_bin, normalize='true')
    print("\nConfusion matrix (normalized):")
    print(cm)

    os.makedirs(args.output_dir, exist_ok=True)
    cm_output_path = os.path.join(args.output_dir, f"confusion_matrix_{args.model_name.replace('/', '_')}_id{args.label_id}.json")
    with open(cm_output_path, "w") as f:
        json.dump(cm.tolist(), f)

    print(f"Confusion matrix saved to {cm_output_path}")

    thresholds = [0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        pred_bin = (test_predictions >= threshold).astype(int)
        true_bin = (test_labels >= threshold).astype(int)

        cm = confusion_matrix(true_bin, pred_bin, normalize='true')
        print(f"\nConfusion matrix (normalized) for threshold {threshold}:")
        print(cm)

        cm_output_path = os.path.join(
            args.output_dir,
            f"confusion_matrix_{args.model_name.replace('/', '_')}_id{args.label_id}_threshold_{threshold}.json"
        )
        with open(cm_output_path, "w") as f:
            json.dump(cm.tolist(), f)

        print(f"Confusion matrix for threshold {threshold} saved to {cm_output_path}")


if __name__ == "__main__":
    main()
