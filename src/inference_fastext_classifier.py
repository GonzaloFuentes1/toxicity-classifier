import argparse

import fasttext
from datasets import concatenate_datasets, load_from_disk

# Variable global para almacenar el modelo fastText ya cargado.
toxicity_classifier = None


def init_model(model_path: str):
    """
    Carga el modelo fastText si aún no está cargado.

    Args:
        model_path (str): Ruta al modelo fastText (archivo .bin).
    """
    global toxicity_classifier
    if toxicity_classifier is None:
        try:
            toxicity_classifier = fasttext.load_model(model_path)
            print(f"Modelo fastText cargado desde: {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error al cargar el modelo fastText desde '{model_path}': {e}"
            )


def classify_toxicity(batch: dict, text_column: str) -> dict:
    """
    Clasifica un batch de textos de la siguiente forma:
    - Se divide cada texto en párrafos usando '\n'.
    - Si alguno de los párrafos se clasifica como '__label__1', se etiqueta
    el texto completo como '__label__1' y se asigna como score el máximo score
    observado.
    - Si ningún párrafo es '__label__1', se asigna '__label__0' y se toma el máximo
    score de cualquier párrafo.

    Args:
        batch (dict): Diccionario con listas de ejemplos; se espera que contenga la
        clave 'text_column'.
        text_column (str): Nombre de la columna de texto en el batch.

    Returns:
        dict: Diccionario con las claves 'toxicity_label' y 'toxicity_score'.
    """
    global toxicity_classifier
    texts = batch.get(text_column, [])
    labels = []
    scores = []

    for text in texts:
        # Separar el texto en párrafos y omitir aquellos vacíos
        paragraphs = [p for p in text.split("\n") if p.strip() != ""]
        if not paragraphs:
            labels.append("__label__0")
            scores.append(0.0)
            continue

        try:
            # Se predice para cada párrafo
            paragraph_preds = [toxicity_classifier.predict(p, k=1) for p in paragraphs]
        except Exception as e:
            raise RuntimeError(f"Error en la predicción de un párrafo: {e}")

        # Determinar si alguno de los párrafos es tóxico
        is_toxic = any(pred[0][0] == "__label__1" for pred in paragraph_preds)
        if is_toxic:
            labels.append("__label__1")
            toxic_scores = [
                pred[1][0] for pred in paragraph_preds if pred[0][0] == "__label__1"
            ]
            scores.append(max(toxic_scores) if toxic_scores else 0.0)
        else:
            labels.append("__label__0")
            all_scores = [pred[1][0] for pred in paragraph_preds]
            scores.append(max(all_scores) if all_scores else 0.0)

    return {"toxicity_label": labels, "toxicity_score": scores}


def process_batch(batch: dict, model_path: str, text_column: str) -> dict:
    """
    Inicializa el modelo fastText (si no se ha hecho aún) y clasifica el batch.

    Args:
        batch (dict): Batch a procesar.
        model_path (str): Ruta al modelo fastText.
        text_column (str): Nombre de la columna de texto.

    Returns:
        dict: Diccionario con las predicciones.
    """
    init_model(model_path)
    return classify_toxicity(batch, text_column)


def print_model_hyperparameters(model_path: str):
    """
    Imprime los hiperparámetros del modelo fastText.

    Args:
        model_path (str): Ruta al modelo fastText.
    """
    try:
        model = fasttext.load_model(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Error al cargar el modelo para imprimir hiperparámetros: {e}"
        )

    print("Hiperparámetros del modelo:")
    print(f"Dimensión de los vectores: {model.get_dimension()}")
    print(f"Cantidad de palabras: {len(model.get_words())}")
    print(f"Cantidad de etiquetas: {len(model.get_labels())}")
    try:
        args_model = model.f.getArgs()
    except Exception as e:
        raise RuntimeError(f"Error al obtener los argumentos del modelo: {e}")
    print(f"Aprendizaje: {args_model.lr}")
    print(f"Cantidad de épocas: {args_model.epoch}")
    print(f"Tamaño del batch: {args_model.ws}")
    print(f"Negativas: {args_model.neg}")
    print(f"Perdida: {args_model.loss}")
    print(f"Submuestreo: {args_model.t}")
    print(f"Tamaño del n-grama: {args_model.wordNgrams}")
    print(f"Longitud mínima de subpalabras (minn): {args_model.minn}")
    print(f"Longitud máxima de subpalabras (maxn): {args_model.maxn}")
    print(f"Buckets para subpalabras: {args_model.bucket}")


def main(args):
    """
    Función principal para procesar el dataset en fragmentos utilizando fastText.

    Args:
        args: Argumentos parseados desde la línea de comandos.
    """
    model_path = args.model_path
    dataset_path = args.dataset_path
    output_path = args.output_path
    text_column = args.text_column
    batch_size = args.batch_size
    chunk_size = args.chunk_size

    # Cargar dataset y validar existencia
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        raise RuntimeError(f"Error al cargar el dataset desde '{dataset_path}': {e}")

    total_examples = len(dataset)
    if total_examples == 0:
        raise ValueError("El dataset está vacío.")

    print(f"Dataset cargado con {total_examples} ejemplos.")

    # Lista para almacenar fragmentos procesados
    processed_chunks = []

    # Procesar el dataset en fragmentos
    for start_idx in range(0, total_examples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_examples)
        print(f"Procesando fragmento {start_idx} - {end_idx} ...")
        try:
            chunk = dataset.select(range(start_idx, end_idx))
        except Exception as e:
            raise RuntimeError(
                f"Error al seleccionar fragmento {start_idx}-{end_idx}: {e}"
            )

        try:
            results = chunk.map(
                lambda batch: process_batch(batch, model_path, text_column),
                batched=True,
                batch_size=batch_size,
                num_proc=1,  # Procesa en CPU; se puede ajustar según los recursos
            )
        except Exception as e:
            raise RuntimeError(
                f"Error al procesar el fragmento {start_idx}-{end_idx}: {e}"
            )

        processed_chunks.append(results)
        print(f"Fragmento procesado: {start_idx} - {end_idx}")

    # Combinar todos los fragmentos en un solo dataset
    try:
        combined_dataset = concatenate_datasets(processed_chunks)
    except Exception as e:
        raise RuntimeError(f"Error al combinar fragmentos procesados: {e}")

    try:
        combined_dataset.save_to_disk(output_path)
    except Exception as e:
        raise RuntimeError(
            f"Error al guardar el dataset etiquetado en '{output_path}': {e}"
        )

    print(f"Dataset combinado guardado en: {output_path}")
    print("Procesamiento completo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clasificación de toxicidad por párrafo usando fastText (CPU)."
    )
    parser.add_argument(
        "--model_path", required=True, help="Ruta al modelo fastText (.bin)"
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Ruta al dataset por etiquetar."
    )
    parser.add_argument(
        "--output_path", required=True, help="Ruta para guardar el dataset etiquetado."
    )
    parser.add_argument(
        "--text_column",
        required=True,
        help="Nombre de la columna de texto del dataset.",
    )
    parser.add_argument(
        "--batch_size", required=True, type=int, help="Tamaño del batch."
    )
    parser.add_argument(
        "--chunk_size",
        required=True,
        type=int,
        help="Número de ejemplos por fragmento.",
    )
    args = parser.parse_args()

    # Imprimir hiperparámetros del modelo para verificación
    print_model_hyperparameters(args.model_path)

    # Ejecutar la clasificación y procesamiento del dataset usando los argumentos
    main(args)
