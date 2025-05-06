import argparse
import os

from datasets import load_from_disk
from transformers import AutoTokenizer


def split_paragraphs(dataset_path: str, text_column: str, expanded_path: str):
    """
    Divide los textos en párrafos y los guarda en un nuevo dataset.
    Args:
        dataset_path: Ruta al dataset original
        text_column: Nombre de la columna que contiene el texto
        expanded_path: Ruta donde guardar el dataset expandido
    """
    ds = load_from_disk(dataset_path)

    def _split(examples, indices):
        texts, parents = [], []
        for txt, idx in zip(examples[text_column], indices):
            pars = [
                p.strip() for p in txt.split("\n") if p.strip() and len(p.strip()) > 30
            ][:10]
            texts.extend(pars)
            parents.extend([idx] * len(pars))
        return {"paragraph_text": texts, "parent_id": parents}

    para_ds = ds.map(
        _split,
        batched=True,
        with_indices=True,
        remove_columns=ds.column_names,
        num_proc=max(1, os.cpu_count() // 2),
        desc="Split paragraphs",
    )
    para_ds.save_to_disk(expanded_path)
    print(f"Párrafos guardados en: {expanded_path}")


def tokenize_paragraphs(expanded_path: str, tok_path: str, model_path: str,
                        max_length: int):
    """
    Tokeniza los párrafos utilizando un tokenizador específico.
    Args:
        expanded_path: Ruta al dataset de párrafos
        tok_path: Ruta donde guardar los párrafos tokenizados
        model_path: Ruta al modelo para usar su tokenizador
        max_length: Longitud máxima de secuencia
    """
    para_ds = load_from_disk(expanded_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _tok(batch):
        return tokenizer(
            batch["paragraph_text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tok_ds = para_ds.map(
        _tok,
        batched=True,
        remove_columns=para_ds.column_names,
        num_proc=max(1, os.cpu_count() // 2),
        desc="Tokenizing paragraphs",
    )
    tok_ds.set_format("torch", columns=["input_ids", "attention_mask"])
    tok_ds.save_to_disk(tok_path)
    print(f"Dataset tokenizado guardado en: {tok_path}")


def main():
    """Función principal que procesa los argumentos y ejecuta el flujo completo."""
    parser = argparse.ArgumentParser(description="Expandir y tokenizar párrafos")
    parser.add_argument(
        "--dataset_path", required=True,
        help="Ruta al dataset original en disco"
    )
    parser.add_argument(
        "--text_column", default="text",
        help="Nombre de la columna de texto en el dataset"
    )
    parser.add_argument(
        "--expanded_path", required=True,
        help="Ruta donde guardar párrafos expandido"
    )
    parser.add_argument(
        "--tok_path", required=True,
        help="Ruta donde guardar párrafos tokenizados"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Ruta al modelo HF para tokenización"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Longitud máxima de tokens"
    )
    args = parser.parse_args()

    os.makedirs(args.expanded_path, exist_ok=True)
    os.makedirs(args.tok_path, exist_ok=True)

    split_paragraphs(args.dataset_path, args.text_column, args.expanded_path)
    tokenize_paragraphs(args.expanded_path, args.tok_path, args.model_path,
                        args.max_length)


if __name__ == "__main__":
    main()
