import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def load_data(
    json_path: str, threshold: float, label_id: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga y procesa los datos desde un archivo JSON para preparar
    los DataFrames de entrenamiento, validación y prueba.

    Args:
        json_path (str): Ruta al archivo JSON con los datos.
        threshold (float): Threshold para considerar un texto como tóxico.
        label_id (int): ID de la categoría a evaluar (default=0).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames de
        entrenamiento, validación y prueba.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = list(map(lambda x: x["original_text"], data))
    labels = list(
        map(
            lambda x: 1
            if x["moderation_result"]["moderationCategories"][label_id]["confidence"]
            > threshold
            else 0,
            data,
        )
    )

    df = pd.DataFrame({"text": texts, "label": labels})

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, stratify=test_df["label"], random_state=42
    )

    return train_df, val_df, test_df


def undersample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza undersampling para balancear las clases en un DataFrame,
    reduciendo la clase mayoritaria sin reemplazo.

    Args:
        df (pd.DataFrame): DataFrame con las columnas `text` y `label`.

    Returns:
        pd.DataFrame: DataFrame balanceado.
    """
    class_0 = df[df["label"] == 0]
    class_1 = df[df["label"] == 1]

    if len(class_0) == 0 or len(class_1) == 0:
        print(
            "⚠ No se puede aplicar undersampling ya que alguna clase no tiene muestras."
        )
        return df

    if len(class_0) > len(class_1):
        class_0 = resample(
            class_0, replace=False, n_samples=len(class_1), random_state=42
        )
    else:
        class_1 = resample(
            class_1, replace=False, n_samples=len(class_0), random_state=42
        )

    balanced_df = (
        pd.concat([class_0, class_1])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return balanced_df


def oversample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza oversampling para balancear las clases en un DataFrame,
    aumentando la clase minoritaria mediante remuestreo con reemplazo.

    Args:
        df (pd.DataFrame): DataFrame con las columnas `text` y `label`.

    Returns:
        pd.DataFrame: DataFrame balanceado.
    """
    class_0 = df[df["label"] == 0]
    class_1 = df[df["label"] == 1]

    if len(class_0) == 0 or len(class_1) == 0:
        print(
            "⚠ No se puede aplicar oversampling ya que alguna clase no tiene muestras."
        )
        return df

    if len(class_0) < len(class_1):
        class_0 = resample(
            class_0, replace=True, n_samples=len(class_1), random_state=42
        )
    else:
        class_1 = resample(
            class_1, replace=True, n_samples=len(class_0), random_state=42
        )

    balanced_df = (
        pd.concat([class_0, class_1])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return balanced_df


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Calcula métricas de evaluación para el modelo.

    Args:
        eval_pred (Any): Tuple que contiene logits y etiquetas verdaderas.

    Returns:
        Dict[str, float]: Métricas calculadas (accuracy, precision, recall, f1).
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo de clasificación de toxicidad."
    )
    parser.add_argument(
        "--json_path", required=True, help="Ruta al archivo JSON con los datos."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Ruta para guardar los resultados del modelo.",
    )
    parser.add_argument(
        "--logging_dir",
        required=True,
        help="Ruta para guardar los logs del entrenamiento.",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        help="Threshold para considerar un texto como tóxico.",
    )
    parser.add_argument("--model_name", required=True, help="Modelo a finetunear.")
    parser.add_argument(
        "--label_id",
        required=False,
        default=0,
        type=int,
        help="ID de la categoría a evaluar.",
    )
    parser.add_argument(
        "--undersample",
        action="store_true",
        help="Aplicar undersampling al set de entrenamiento.",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Aplicar oversampling al set de entrenamiento.",
    )
    args = parser.parse_args()

    # Se verifica que no se especifiquen ambas estrategias
    if args.undersample and args.oversample:
        print(
            "Debes elegir solo una estrategia: --undersample o --oversample, no ambas."
        )
        exit(1)

    json_path = args.json_path
    model_name = args.model_name
    output_dir = os.path.join(args.output_dir, f"{model_name}_id{args.label_id}")
    logging_dir = os.path.join(args.logging_dir, f"{model_name}_id{args.label_id}")
    threshold = float(args.threshold)

    train_df, val_df, test_df = load_data(json_path, threshold, args.label_id)

    # Aplicar estrategia de balanceo en el set de entrenamiento según el flag indicado
    if args.undersample:
        print("⚡ Aplicando undersampling al set de entrenamiento")
        balanced_train_df = undersample_dataframe(train_df)
    elif args.oversample:
        print("⚡ Aplicando oversampling al set de entrenamiento")
        balanced_train_df = oversample_dataframe(train_df)
    else:
        print("⚡ Sin balanceo aplicado al set de entrenamiento")
        balanced_train_df = train_df

    # Para validación y prueba no se aplica balanceo
    balanced_val_df = val_df
    balanced_test_df = test_df

    train_dataset = Dataset.from_pandas(balanced_train_df)
    val_dataset = Dataset.from_pandas(balanced_val_df)
    test_dataset = Dataset.from_pandas(balanced_test_df)

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(
        examples: Dict[str, str],
    ) -> Dict[str, Union[List[int], List[str]]]:
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    num_labels = 2

    # Cargar modelo: si se usa un modelo en particular se puede personalizar la carga
    if model_name == "unitary/multilingual-toxic-xlm-roberta":
        base_model = XLMRobertaModel.from_pretrained(
            "unitary/multilingual-toxic-xlm-roberta"
        )
        config = AutoConfig.from_pretrained("unitary/multilingual-toxic-xlm-roberta")
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        model = XLMRobertaForSequenceClassification(config)
        model.roberta.load_state_dict(base_model.state_dict(), strict=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=False,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )

    # Construir un nombre único para el directorio de salida
    balance_method = "no_balance"
    if args.undersample:
        balance_method = "undersample"
    elif args.oversample:
        balance_method = "oversample"
    model_str = args.model_name.replace('/', '_')
    threshold_str = str(args.threshold).replace('.', '_')

    output_dir = os.path.join(
        args.output_dir,
        f"{model_str}_id{args.label_id}_th{threshold_str}_{balance_method}",
    )

    # Crear el directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Actualizar el argumento de TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Evalúa al final de cada época
        save_strategy="epoch",  # Guarda el modelo al final de cada época
        load_best_model_at_end=True,  # Carga el mejor modelo al final del entrenamiento
        metric_for_best_model="eval_f1",  # Métrica para determinar el mejor modelo
        greater_is_better=True,  # Indica que un mayor F1 es mejor
        learning_rate=4e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=logging_dir,
        logging_steps=20,
        gradient_accumulation_steps=1,
        bf16=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluar el modelo en el conjunto de validación
    print("\n--- Validation Set ---")
    val_results = trainer.evaluate()
    print(val_results)

    # Evaluar el modelo en el conjunto de prueba
    print("\n--- Test Set ---")
    test_results = trainer.predict(tokenized_datasets["test"])
    test_predictions = test_results.predictions.argmax(axis=-1)
    test_labels = test_results.label_ids

    # Calcular y guardar la matriz de confusión
    cm = confusion_matrix(test_labels, test_predictions, normalize="true")
    print("\nConfusion matrix (normalized):")
    print(cm)

    cm_output_path = os.path.join(
        output_dir,
        f"confusion_matrix_{args.model_name.replace('/', '_')}_id{args.label_id}.json",
    )
    with open(cm_output_path, "w") as cm_file:
        json.dump(cm.tolist(), cm_file)

    print(f"Confusion matrix saved to {cm_output_path}")


if __name__ == "__main__":
    main()
