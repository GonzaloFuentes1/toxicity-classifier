import argparse
import os

import torch
from datasets import load_from_disk
from transformers import pipeline

# Variable global para almacenar la instancia del pipeline
toxicity_classifier = None


def init_model(model_path: str, device_id: int):
    """
    Inicializa el pipeline de clasificación en la GPU especificada (device_id).
    Se ejecuta solo una vez por proceso, gracias a la verificación de la variable

    Args:
        model_path (str): Ruta al modelo preentrenado.
        device_id (int): ID de la GPU que se utilizará.
    """
    global toxicity_classifier
    if toxicity_classifier is None:
        try:
            toxicity_classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                device=device_id,
                truncation=True,
                max_length=512,
            )
            print(f"Pipeline inicializado en GPU {device_id}")
        except Exception as e:
            raise RuntimeError(
                f"Error al inicializar el pipeline en GPU {device_id}: {e}"
            )


def classify_toxicity(batch: dict, text_column: str, batch_size: int) -> dict:
    """
    Clasifica la toxicidad de un batch de textos utilizando la instancia global
    'toxicity_classifier'.

    Args:
        batch (dict): Batch de ejemplos con al menos la clave 'text_column'.
        text_column (str): Nombre de la columna de texto.
        batch_size (int): Tamaño del batch a procesar.

    Returns:
        dict: Diccionario con las predicciones 'toxicity_label' y 'toxicity_score'.
    """
    global toxicity_classifier
    if toxicity_classifier is None:
        raise RuntimeError("El pipeline aún no ha sido inicializado.")

    texts = batch.get(text_column, None)
    if texts is None:
        raise ValueError(f"La llave '{text_column}' no se encontró en el batch.")

    try:
        results = toxicity_classifier(texts, batch_size=batch_size)
    except Exception as e:
        raise RuntimeError(f"Error durante la clasificación en el batch: {e}")

    labels = [res["label"] for res in results]
    scores = [res["score"] for res in results]
    return {"toxicity_label": labels, "toxicity_score": scores}


def process_batch(
    batch: dict,
    process_index: int,
    model_path: str,
    text_column: str,
    batch_size: int,
    num_gpus: int,
) -> dict:
    """
    Función a ejecutar en paralelo mediante dataset.map().
    Se encarga de calcular el ID de la GPU correspondiente según el 'process_index',
    inicializar el modelo (sólo una vez por proceso) y clasificar la toxicidad delbatch

    Args:
        batch (dict): Batch de ejemplos a procesar.
        process_index (int): Índice del proceso en ejecución.
        model_path (str): Ruta al modelo preentrenado.
        text_column (str): Nombre de la columna de texto.
        batch_size (int): Tamaño del batch.
        num_gpus (int): Número total de GPUs disponibles.

    Returns:
        dict: Diccionario con las etiquetas y scores de toxicidad.
    """
    device_id = process_index % num_gpus
    init_model(model_path, device_id)
    return classify_toxicity(batch, text_column, batch_size)


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Clasificación de toxicidad en paralelo usando GPUs."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo de clasificación de toxicidad (pipeline de Hugging Face).",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Ruta al dataset (almacenado en disco) que se desea etiquetar.",
    )
    parser.add_argument(
        "--output_path", required=True, help="Ruta para guardar el dataset etiquetado."
    )
    parser.add_argument(
        "--text_column",
        required=True,
        help="Nombre de la columna que contiene el texto en el dataset.",
    )
    parser.add_argument(
        "--batch_size", required=True, type=int, help="Tamaño del batch a procesar."
    )
    args = parser.parse_args()

    # Verificar existencia y permisos del dataset de entrada
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"El dataset no se encontró en: {args.dataset_path}")

    print("Cargando dataset desde disco...")
    try:
        dataset = load_from_disk(args.dataset_path)
    except Exception as e:
        raise RuntimeError(f"Error al cargar el dataset: {e}")

    # Filtrado del dataset según la etiqueta de toxicidad preexistente
    try:
        dataset = dataset.filter(
            lambda example: example.get("toxicity_prediction", None) == "__label__1",
            num_proc=128,
            desc="Filtering toxic tweets",
        )
    except Exception as e:
        raise RuntimeError(f"Error durante el filtrado del dataset: {e}")

    print("Dataset cargado y filtrado.")

    # Detectar GPUs disponibles
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No se encontraron GPUs disponibles.")
    else:
        print(f"Se encontraron {num_gpus} GPU(s).")

    # Procesamiento paralelo del dataset
    try:
        results = dataset.map(
            lambda batch, process_index: process_batch(
                batch,
                process_index,
                args.model_path,
                args.text_column,
                args.batch_size,
                num_gpus,
            ),
            batched=True,
            batch_size=args.batch_size,
            num_proc=num_gpus,
            with_rank=True,
        )
    except Exception as e:
        raise RuntimeError(f"Error durante el procesamiento paralelo del dataset: {e}")

    try:
        results.save_to_disk(args.output_path)
    except Exception as e:
        raise RuntimeError(
            f"Error al guardar el dataset etiquetado en {args.output_path}: {e}"
        )

    print(f"Dataset etiquetado guardado en: {args.output_path}")


if __name__ == "__main__":
    main()
