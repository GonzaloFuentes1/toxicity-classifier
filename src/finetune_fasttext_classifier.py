import argparse
import json

import fasttext
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def load_data(
    json_path: str, threshold: float, label: str, balance_method: str = None
) -> None:
    """
    Carga los datos desde un archivo JSON y los prepara en archivos de texto
    para ser utilizados por fastText, pudiendo balancear las clases en el
    conjunto de entrenamiento mediante undersampling u oversampling.

    Args:
        json_path (str): Ruta al archivo JSON con los datos.
        threshold (float): Threshold para considerar un texto como tóxico.
        label (str): Nombre de la categoría a evaluar.
        balance_method (str, optional): Estrategia de balanceo a aplicar en el set de
                                entrenamiento. Puede ser "undersample", "oversample"
                                o None (sin balanceo). Por defecto es None.
    """
    # Cargar JSON
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extraer textos y asignar etiquetas según el threshold
    texts = [x["original_text"] for x in data]
    labels = []
    for x in data:
        # Buscar la categoría indicada
        category = next(
            (
                cat
                for cat in x["moderation_result"]["moderationCategories"]
                if cat["name"] == label
            ),
            None,
        )
        confidence = category["confidence"] if category else 0.0
        labels.append("__label__1" if confidence > threshold else "__label__0")

    df = pd.DataFrame({"text": texts, "label": labels})

    # Dividir los datos en entrenamiento, validación y prueba
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    # Balancear el conjunto de entrenamiento si se especifica una estrategia
    if balance_method is not None:
        class_0 = train_df[train_df["label"] == "__label__0"]
        class_1 = train_df[train_df["label"] == "__label__1"]

        if balance_method == "undersample":
            # Si alguna clase no tiene muestras, se mantiene el set original
            if len(class_0) == 0 or len(class_1) == 0:
                print("No se puede realizar undersampling porque alguna clase")
            else:
                # Remuestreo sin reemplazo de la clase mayoritaria
                if len(class_0) > len(class_1):
                    class_0 = resample(
                        class_0, replace=False, n_samples=len(class_1), random_state=42
                    )
                else:
                    class_1 = resample(
                        class_1, replace=False, n_samples=len(class_0), random_state=42
                    )
        elif balance_method == "oversample":
            # Si alguna clase no tiene muestras, se mantiene el set original
            if len(class_0) == 0 or len(class_1) == 0:
                print("No se puede realizar oversampling porque alguna clase no tiene")
            else:
                # Remuestreo con reemplazo de la clase minoritaria
                if len(class_0) < len(class_1):
                    class_0 = resample(
                        class_0, replace=True, n_samples=len(class_1), random_state=42
                    )
                else:
                    class_1 = resample(
                        class_1, replace=True, n_samples=len(class_0), random_state=42
                    )
        else:
            print(
                f"⚠ Estrategia de balanceo '{balance_method}'desconocida. Se procederá",
                "sin balanceo.",
            )

        train_df = (
            pd.concat([class_0, class_1])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    # Función para guardar en formato fastText
    def save_fasttext_data(df, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                # Limpiar saltos de línea y espacios múltiples
                text = row["text"].replace("\n", " ").replace("\r", " ").strip()
                f.write(f"{row['label']} {text}\n")

    # Guardar archivos para fastText
    save_fasttext_data(train_df, "train.txt")
    save_fasttext_data(val_df, "valid.txt")
    save_fasttext_data(test_df, "test.txt")


def evaluate_model(model: fasttext.FastText, test_file: str, model_name: str):
    """
    Evalúa el modelo en un conjunto de prueba y guarda las métricas en un archivo.

    Args:
        model (fasttext.FastText): El modelo entrenado.
        test_file (str): Archivo de texto con los datos de prueba.
        model_name (str): Nombre (o ruta) del modelo, nombrar el archivo de métricas.
    """
    # Leer el archivo de prueba
    with open(test_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split(" ", 1) for line in f if line.strip()]

    # Crear DataFrame a partir de los datos
    test_df = pd.DataFrame(lines, columns=["label", "text"])

    # Convertir etiquetas a numéricas para evaluar
    labels_true = [1 if lbl == "__label__1" else 0 for lbl in test_df["label"]]
    labels_pred = [
        1 if model.predict(txt)[0][0] == "__label__1" else 0 for txt in test_df["text"]
    ]

    accuracy = accuracy_score(labels_true, labels_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_true, labels_pred, average="binary"
    )
    conf_matrix = confusion_matrix(labels_true, labels_pred)

    metrics_text = (
        f"Accuracy: {accuracy: .4f}\n"
        f"Precision: {precision: .4f}\n"
        f"Recall: {recall: .4f}\n"
        f"F1-score: {f1: .4f}\n"
        f"Confusion Matrix: \n{conf_matrix}\n"
    )
    print(metrics_text)

    with open(f"{model_name}_metrics.txt", "w", encoding="utf-8") as metrics_file:
        metrics_file.write(metrics_text)


def main():
    parser = argparse.ArgumentParser(
        description="FastText toxicity classifier trainer with optional autotune."
    )
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--autotuneDuration", type=int, default=0)

    parser.add_argument(
        "--word_ngrams", type=int, help="Set n-grams manually (optional)"
    )
    parser.add_argument("--epoch", type=int, help="Set epochs manually (optional)")
    parser.add_argument(
        "--lr", type=float, help="Set learning rate manually (optional)"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--undersample", action="store_true")
    group.add_argument("--oversample", action="store_true")

    args = parser.parse_args()

    balance_method = (
        "undersample" if args.undersample else "oversample" if args.oversample else None
    )

    load_data(args.json_path, args.threshold, args.label, balance_method)

    print("Entrenando modelo...")
    train_args = {"input": "train.txt"}

    if args.autotuneDuration > 0:
        train_args["autotuneValidationFile"] = "valid.txt"
        train_args["autotuneDuration"] = args.autotuneDuration

    # Si el usuario fija algún hiperparámetro, fastText lo respetará incluso con
    if args.word_ngrams is not None:
        train_args["wordNgrams"] = args.word_ngrams
    if args.epoch is not None:
        train_args["epoch"] = args.epoch
    if args.lr is not None:
        train_args["lr"] = args.lr

    model = fasttext.train_supervised(**train_args)
    model.save_model(args.output_model)

    print("Evaluando modelo...")
    evaluate_model(model, "test.txt", args.output_model)


if __name__ == "__main__":
    main()
