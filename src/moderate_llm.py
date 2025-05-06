import argparse
import json
import os
from collections import defaultdict

from datasets import load_from_disk
from huggingface_hub import login, snapshot_download
from tqdm import tqdm
from vllm import LLM, SamplingParams

categories = ["Toxic", "Insult", "Profanity", "Sexual", "Violent"]

login("")


def get_prompt(model_name, text, prompt_template):
    formatted_text = prompt_template.format(
        text=text.replace("\n", " ").strip()[:2500]
    )
    if any(x in model_name.lower() for x in ["mixtral", "mistral"]):
        return f"<s>[INST] {formatted_text} [/INST]"
    elif "llama" in model_name.lower():
        return (
            f"<|begin_of_text|><|user|>\n{formatted_text}"
            f"\n<|end_of_text|><|assistant|>"
        )
    elif "deepseek" in model_name.lower():
        return f"[Instrucción]\n{formatted_text}\n[Respuesta]"
    else:
        return formatted_text


def ensure_model_downloaded(repo_id: str, local_path: str):
    if (
        not os.path.exists(local_path)
        or not os.path.exists(os.path.join(local_path, "config.json"))
    ):
        print(f"⬇️ Descargando modelo {repo_id} en {local_path}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"✅ Modelo ya presente en {local_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Ruta al archivo .txt que contiene el prompt base",
    )
    parser.add_argument("--download_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--text_column", default="original_text")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cuda_devices", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.makedirs("partial_outputs", exist_ok=True)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    ds = load_from_disk(args.dataset_path)
    if args.text_column not in ds.column_names:
        raise ValueError(
            f"La columna '{args.text_column}' no se encuentra en el dataset."
        )
    texts = ds[args.text_column][:100000000]

    texts_full = []
    for text in tqdm(texts):
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            if len(paragraph) > 30:
                texts_full.append(paragraph)
    texts = texts_full

    model_preds = {model_name: [] for model_name in args.model_names}

    for model_name in args.model_names:
        print(f"\n🔍 Procesando modelo: {model_name}")
        local_path = os.path.join(
            args.download_dir, model_name.replace("/", "_")
        )
        ensure_model_downloaded(model_name, local_path)

        llm = LLM(
            model=model_name,
            download_dir=args.download_dir,
            tensor_parallel_size=len(args.cuda_devices.split(",")),
            enable_prefix_caching=True,
            max_model_len=4096,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=4096,
            truncate_prompt_tokens=4095,
        )

        partial_dir = os.path.join(
            "partial_outputs", model_name.replace("/", "_")
        )
        os.makedirs(partial_dir, exist_ok=True)
        completed_batches = set(
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(partial_dir)
            if f.endswith(".jsonl")
        )

        for idx in tqdm(
            range(0, len(texts), args.batch_size),
            desc=f"Procesando {model_name}",
        ):
            batch_id = idx // args.batch_size
            if batch_id in completed_batches:
                continue

            batch = texts[idx: idx + args.batch_size]
            prompts = [
                get_prompt(model_name, t, prompt_template) for t in batch
            ]
            outputs = llm.generate(prompts, sampling_params)

            batch_results = []
            error = 0
            for i, output in enumerate(outputs):
                try:
                    raw_output = output.outputs[0].text.strip()
                    print("🔍 Salida cruda del modelo:\n", raw_output)
                    if not raw_output:
                        error += 1
                        raise ValueError("Modelo devolvió una cadena vacía")
                    parsed_json = json.loads(raw_output)
                    parsed = {
                        cat: min(
                            1.0,
                            max(
                                0.0,
                                float(
                                    parsed_json.get(cat, {}).get("score", 0.0)
                                ),
                            ),
                        )
                        for cat in categories
                    }
                except Exception as e:
                    print(
                        f"⚠️ Error en batch {batch_id}, modelo {model_name}: {e}"
                    )
                    continue
                model_preds[model_name].append(parsed)
                batch_results.append(
                    json.dumps(
                        {"input": batch[i], "prediction": parsed},
                        ensure_ascii=False,
                    )
                )

            print(error)
            with open(
                os.path.join(partial_dir, f"batch_{batch_id}.jsonl"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(batch_results))

    results = []
    for i in range(len(texts)):
        acc = defaultdict(float)
        for model_name in args.model_names:
            acc_model = model_preds[model_name][i]
            for cat in categories:
                acc[cat] += acc_model.get(cat, 0.0)
        avg = {
            cat: round(acc[cat] / len(args.model_names), 4)
            for cat in categories
        }
        results.append({"input": texts[i], "avg_prediction": avg})

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"\n✅ Clasificación completada. Resultados finales guardados en:"
        f" {args.output_path}"
    )