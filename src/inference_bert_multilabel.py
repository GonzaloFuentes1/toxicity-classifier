import argparse
import os
import time

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

os.environ["WANDB_MODE"] = "disabled"


def group_by_parent(
    orig_path: str,
    expanded_path: str,
    scores: np.ndarray,
    threshold: float,
    output_path: str,
    shard_id: int,
    num_shards: int,
):
    from collections import defaultdict

    print(f"[INFO] Cargando datasets para shard {shard_id}/{num_shards}...")
    t0 = time.time()
    orig_ds = load_from_disk(orig_path)

    para_ds = load_from_disk(expanded_path)
    if num_shards > 1:
        para_ds = para_ds.shard(num_shards=num_shards, index=shard_id)
    print(f"[INFO] Párrafos cargados en {time.time() - t0: .2f} s")

    # Añadir scores y agrupar por parent_id
    preds = para_ds.add_column("toxicity_scores", scores.tolist())
    grouped = defaultdict(list)
    for pid, sc in zip(preds["parent_id"], preds["toxicity_scores"]):
        grouped[pid].append(sc)

    parent_ids = sorted(grouped.keys())
    print(f"[INFO] Este shard cubre {len(parent_ids)} textos originales")

    # Extraer de golpe sólo esos textos del original
    subset = orig_ds.select(parent_ids)

    # Calcular listas de máximos y etiquetas
    max_scores_list = []
    max_labels_list = []
    any_toxic_list = []
    for pid in parent_ids:
        arr = np.vstack(grouped[pid])        # (#párrafos, #clases)
        max_sc = arr.max(axis=0).tolist()    # [score_clase0, clase1, ...]
        labels = [1 if s >= threshold else 0 for s in max_sc]
        any_t = int(any(labels))
        max_scores_list.append(max_sc)
        max_labels_list.append(labels)
        any_toxic_list.append(any_t)

    # Añadir las tres columnas de golpe
    subset = subset.add_column("toxicity_max_scores", max_scores_list)
    subset = subset.add_column("toxicity_max_labels", max_labels_list)
    subset = subset.add_column("any_toxic_label", any_toxic_list)

    # Guardar shard parcial
    shard_path = f"{output_path}_shard{shard_id}"
    subset.save_to_disk(shard_path)
    info = f"[INFO] Shard {shard_id} guardado con {len(parent_ids)} ejemplos"
    print(f"{info} en: {shard_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia multigpu por shards con Trainer.predict"
    )
    parser.add_argument("--tok_path", required=True, help="Dataset tokenizado")
    parser.add_argument("--orig_path", required=True, help="Dataset original")
    parser.add_argument("--expanded_path", required=True, help="Dataset expandido")
    parser.add_argument("--model_path", required=True, help="Checkpoint HF")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral")
    parser.add_argument("--output_path", required=True, help="Prefijo de salida")
    parser.add_argument("--shard_id", type=int, default=0, help="ID de shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards")
    args = parser.parse_args()

    # Carga tokenizados y aplica shard
    tok_ds = load_from_disk(args.tok_path)
    if args.num_shards > 1:
        tok_ds = tok_ds.shard(num_shards=args.num_shards, index=args.shard_id)

    # Modelo y trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_eval_batch_size=args.batch_size,
        fp16=True, dataloader_drop_last=False, report_to=[]
    )
    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

    print(f"🔍 Ejecutando Trainer.predict para shard {args.shard_id}...")
    preds = trainer.predict(tok_ds)
    logits = preds.predictions
    print(f"[INFO] Logits shape: {logits.shape}")

    scores = torch.sigmoid(torch.from_numpy(logits)).numpy()

    # Agrupar y guardar
    group_by_parent(
        args.orig_path,
        args.expanded_path,
        scores,
        args.threshold,
        args.output_path,
        args.shard_id,
        args.num_shards,
    )


if __name__ == "__main__":
    main()
